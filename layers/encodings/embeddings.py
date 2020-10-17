
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from utils import *
from functions import *

from layers.indexings import *
from layers.encodings.seqs import *


class PositionalEmbedding(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        self.pos_embedding = nn.Embedding(512, input_dim)
        init_embedding(self.pos_embedding.weight)
        
    def forward(self, inputs, mask=None):
        
        pos = torch.arange(1, inputs.shape[1]+1)[None].repeat(inputs.shape[0], 1) # (B, T)
        pos = pos.to(inputs.device)
        if mask is not None:
            pos = pos * mask.long() # (B, T)
        
        return inputs + self.pos_embedding(pos)

class CharLSTMEncoding(nn.Module):
    
    def __init__(self, config, input_dim=None, num_layers=1):
        super().__init__()
        self.lstm = LSTMEncoding(config, input_dim, num_layers)
        
    def forward(self, inputs, lens=None, mask=None):
        bs, t_maxlen, c_maxlen, _ = inputs.shape
        inputs = inputs.view(bs*t_maxlen, c_maxlen, -1)
        lens = lens.view(bs*t_maxlen)
        
        _, hids = self.lstm(inputs, return_cls=True, lens=lens, mask=mask) # (bs*t_maxlen, hidden_dim)
        hids = hids.view(bs, t_maxlen, -1)
        return hids
    
class CharCNNEncoding(nn.Module):
    
    def __init__(self, config, input_dim=None):
        super().__init__()
        self.cnn = CNNEncoding(config, input_dim, Ks=[3])
        
    def forward(self, inputs, lens=None, mask=None):
        bs, t_maxlen, c_maxlen, _ = inputs.shape
        inputs = inputs.view(bs*t_maxlen, c_maxlen, -1)
        
        hids = self.cnn(inputs, mask=mask) # (bs*t_maxlen, hidden_dim)
        hids = hids.view(bs, t_maxlen, -1)
        return hids
    
    
class AllEmbedding(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.char_indexing = CharIndexing(cased=True)
        self.token_indexing = Indexing(
            maxlen=self.config.maxlen, vocab_file=self.config.vocab_file, 
            cased=self.config.cased)
        if self.config.vocab_file is None:
            self.token_indexing.vocab = {
                '[PAD]': 0,
                '[MASK]': 1,
                '[CLS]': 2,
                '[SEP]': 3,
                '[UNK]': 4,
            }
        else:
            self.config.vocab_size = len(self.token_indexing.vocab)
    
        self.masking = Masking()
        
        if self.config.char_emb_dim > 0:
            self.char_embedding = nn.Embedding(1000, self.config.char_emb_dim)
            # init
            init_embedding(self.char_embedding.weight)
            _config = config.copy()
            _config.hidden_dim = self.config.char_emb_dim
            if self.config.char_encoder.lower() == 'cnn':
                self.char_encoding = CharCNNEncoding(_config, self.config.char_emb_dim)
            else: # self.config.char_encoder.lower() == 'lstm':
                self.char_encoding = CharLSTMEncoding(_config, self.config.char_emb_dim)
                
        if self.config.lm_emb_dim > 0:
            self.lm_embedding = PreEmbeddedLM(self.config)
                
        if self.config.token_emb_dim > 0:
            self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.token_emb_dim)
            init_embedding(self.token_embedding.weight)
        
        self.char_emb_dropout = nn.Dropout(config.dropout)
        
        self.device = self.config.device
        self.to(self.device)
        
    def load_pretrained(self, path, freeze=True):
        embedding_matrix = self.token_embedding.cpu().weight.data
        with open(path, 'r') as f:
            for line in tqdm(f):
                line = line.strip().split(' ')
                token = line[0]
                vector = np.array([float(x) for x in line[1:]], dtype=np.float32)
                vector = torch.from_numpy(vector)
                idx = self.token_indexing.token2idx(token)
                embedding_matrix[idx] = vector
        self.token_embedding.weight.data = embedding_matrix.to(self.config.device)
        
        if freeze:
            def _freeze_word_embs(self, grad_in, grad_out):
                embs_grad = grad_in[0]
                embs_grad[5:] = 0.
                return (embs_grad,)
            self.token_embedding.register_backward_hook(_freeze_word_embs)
        
    def preprocess_sentences(self, sentences):
        
        if len(sentences) > 1 and isinstance(sentences[1], torch.Tensor):
            # sentences is Tensor or is a list/tuple of Tensor
            return [(x.to(self.device) if isinstance(x, torch.Tensor) else x) for x in sentences]
        
        t_indexs = self.token_indexing(sentences)
        if self.config.char_emb_dim > 0:
            if self.config.char_encoder.lower() == 'cnn':
                c_indexs = self.char_indexing(sentences, pad_position='both')
            else:
                c_indexs = self.char_indexing(sentences)
            return [sentences, t_indexs, c_indexs]
        else:
            return [sentences, t_indexs]
        
    def forward(self, sentences, return_list=False, return_dict=False):
        
        embeddings_list = []
        embeddings_dict = {
            'sentences': sentences,
        }
        
        if self.config.char_emb_dim > 0:
            sentences, t_indexs, c_indexs = self.preprocess_sentences(sentences)
            t_indexs = t_indexs.to(self.device)
            c_indexs = c_indexs.to(self.device)
        else:
            sentences, t_indexs, = self.preprocess_sentences(sentences)
            t_indexs = t_indexs.to(self.device)

        # token
        if self.config.token_emb_dim > 0:
            tmp = self.token_embedding(t_indexs)
            embeddings_list.append(tmp)  
            embeddings_dict['token_emb'] = tmp
            
        masks = self.masking(t_indexs, mask_val=0)
        embeddings_dict['masks'] = masks
        
        # char
        if self.config.char_emb_dim > 0:
            c_masks = self.masking(c_indexs, mask_val=0)
            c_lens = c_masks.sum(dim=-1) + (1-masks.long())
            c_embeddings = self.char_embedding(c_indexs)
            c_embeddings = self.char_emb_dropout(c_embeddings)
            c_embeddings = self.char_encoding(c_embeddings, lens=c_lens, mask=c_masks)
            embeddings_list.append(c_embeddings)
            embeddings_dict['char_emb'] = c_embeddings
            
        # lm
        if self.config.lm_emb_dim > 0:
            lm_embs, lm_heads = self.lm_embedding(sentences)
            embeddings_list.append(lm_embs)
            embeddings_dict['lm_emb'] = lm_embs
            if lm_heads is not None:
                embeddings_dict['lm_heads'] = lm_heads
            
        rets = []
            
        if return_list:
            rets += [embeddings_list, masks]
        else:
            embeddings = torch.cat(embeddings_list, dim=-1)
            embeddings_dict['embs'] = embeddings
            rets += [embeddings, masks]
        
        if return_dict:
            rets += [embeddings_dict]
    
        return rets