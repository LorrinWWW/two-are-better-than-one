
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
from layers.encodings import *


class LSTMEncoding(nn.Module):
    
    def __init__(self, config, input_dim=None, num_layers=1):
        super().__init__()
        self.config = config
        
        self.hidden_dim = config.hidden_dim
        self.num_layers = num_layers
        self.bias = True #config.bias
        self.batch_first = True #config.batch_first
        self.dropout = config.dropout
        self.bidirectional = True #config.bidirectional
        self.input_dim = self.hidden_dim if input_dim is None else input_dim
        
        k_bidirectional = 2 if self.bidirectional else 1
        
        self.lstm = nn.LSTM(
            self.input_dim, self.hidden_dim//k_bidirectional, 
            self.num_layers, self.bias, self.batch_first, self.dropout, self.bidirectional)
        init_lstm(self.lstm)
        
        shape = [k_bidirectional*self.num_layers, 1, self.hidden_dim//k_bidirectional]
    
    def forward(self, inputs, return_cls=False, mask=None, lens=None):
        batch_size = inputs.shape[0] if self.batch_first else input.shape[1]
        hidden = None
        
        if mask is not None or lens is not None:
            if lens is not None:
                word_seq_lens = lens
            else:
                word_seq_lens = mask.sum(dim=-1)
            word_rep = inputs
            sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
            _, recover_idx = permIdx.sort(0, descending=False)
            sorted_seq_tensor = word_rep[permIdx]
            
            packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
            lstm_out, (h, _) = self.lstm(packed_words, None)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            outputs = lstm_out[recover_idx]
            hidden = torch.cat([h[-2,:,:], h[-1,:,:]], dim=-1)
            hidden = hidden[recover_idx]
        else:
            outputs, (h, c) = self.lstm(inputs, hidden)
            hidden = torch.cat([h[-2,:,:], h[-1,:,:]], dim=-1)
        
        if return_cls:
            return (outputs, hidden)
        else:
            return outputs
        
class CNNEncoding(nn.Module):
    ''' n to 1 '''
    def __init__(self, config, input_dim=None, Ks=[3,4,5]):
        super().__init__()
        self.config = config
        
        if input_dim is None:
            input_dim = config.hidden_dim
        
        D = input_dim      # input dim
        C = config.hidden_dim # output dim
        Ci = 1
        Co = config.hidden_dim # kernel numbers i.e. hid dim
        Ks = Ks # kernels
        
        self.convs1 = nn.ModuleList([
            nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), padding=((K-1)//2,0)) for K in Ks])

    def forward(self, x, mask=None, lens=None):
        # x (N, W, D)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [conv(x).squeeze(3) for conv in self.convs1]
        if mask is not None:
            mask = mask.view(x[0].shape[0], 1, -1).float() # (N, 1, W) c.f. x:(N, D, W)
            x = [i - (1.-mask)*999 for i in x]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x
    
class TransformerEncoding(nn.Module):
    
    def __init__(self, config, nhead=4, num_layers=2):
        super().__init__()
        self.config = config
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_dim, nhead=nhead)
        norm = nn.LayerNorm(self.config.hidden_dim)
        self.attn = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm)
        
    def forward(self, inputs, mask=None):
        inputs = inputs.permute(1,0,2)
        src_key_padding_mask = None if mask is None else ~mask
        outputs = self.attn(inputs, src_key_padding_mask=src_key_padding_mask)
        outputs = outputs.permute(1,0,2)
        return outputs
    
class AttentionEncoding(nn.Module):
    ''' n to 1 '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = nn.Linear(config.hidden_dim, 1)
        init_linear(self.attention)
        
    def forward(self, inputs, mask=None):
        a = self.attention(inputs) # (B, T, H) => (B, T, 1)
        if mask is not None:
            a -= 999*(~mask).float()[:, :, None]
        a = F.softmax(a, dim=1) # (B, T, 1)
        outputs = (a*inputs).sum(1) # (B, H)
        return outputs
    
    
class RouterEncoding(nn.Module):
    
    def __init__(self, config, n_iter=2, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
            
        self.config = config
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            SelfRouting(self.config, config.hidden_dim, n_iter=n_iter) for i in range(num_layers)
        ])
        
        self.fcs = nn.ModuleList([
             nn.Sequential(
                 nn.Linear(config.hidden_dim, dim_feedforward),
                 nn.ReLU(),
                 nn.Dropout(dropout),
                 nn.Linear(dim_feedforward, config.hidden_dim),
             ) for i in range(num_layers)
        ])
        
        self.norms0 = nn.ModuleList([
            nn.LayerNorm(self.config.hidden_dim) for _ in range(num_layers)
        ])
        
        self.norms1 = nn.ModuleList([
            nn.LayerNorm(self.config.hidden_dim) for _ in range(num_layers)
        ])
        
        self.scales = nn.Parameter(torch.ones(num_layers).float())
        
    def forward(self, inputs, mask=None):
        outputs = inputs
        for i in range(self.num_layers):
            outputs = outputs + self.dropout(self.scales[i] * self.layers[i](outputs, mask)[0])
            outputs = self.norms0[i](outputs)
            
            outputs = outputs + self.dropout(self.fcs[i](outputs))
            outputs = self.norms1[i](outputs)
            
        return outputs
        
        
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
            maxlen=self.config.maxlen, vocab_file=self.config.vocab_file, cased=self.config.cased)
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
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.token_emb_dim)
        init_embedding(self.token_embedding.weight)
        
        self.char_emb_dropout = nn.Dropout(config.dropout)
        
        self.device = self.config.device
        self.to(self.device)
        
    def load_pretrained(self, path):
        embedding_matrix = self.token_embedding.cpu().weight.data
        with open(path, 'r') as f:
            for line in tqdm(f):
                line = line.strip().split()
                token = line[0]
                vector = np.array([float(x) for x in line[1:]], dtype=np.float32)
                vector = torch.from_numpy(vector)
                idx = self.token_indexing.token2idx(token)
                embedding_matrix[idx] = vector
        self.token_embedding.weight.data = embedding_matrix.to(self.config.device)
        
    def preprocess_sentences(self, sentences):
        
        if isinstance(sentences[0], torch.Tensor):
            # sentences is Tensor or is a list/tuple of Tensor
            return [x.to(self.device) for x in sentences]
        
        t_indexs = self.token_indexing(sentences)
        if self.config.char_emb_dim > 0:
            if self.config.char_encoder.lower() == 'cnn':
                c_indexs = self.char_indexing(sentences, pad_position='both')
            else:
                c_indexs = self.char_indexing(sentences)
            return [t_indexs, c_indexs]
        else:
            return [t_indexs]
        
    def forward(self, sentences):
        
        if self.config.char_emb_dim > 0:
            t_indexs, c_indexs = self.preprocess_sentences(sentences)
            t_indexs = t_indexs.to(self.device)
            c_indexs = c_indexs.to(self.device)
            masks = self.masking(t_indexs, mask_val=0)
            c_masks = self.masking(c_indexs, mask_val=0)
            c_lens = c_masks.sum(dim=-1) + (1-masks.long())
            t_embeddings = self.token_embedding(t_indexs)
            c_embeddings = self.char_embedding(c_indexs)
            c_embeddings = self.char_emb_dropout(c_embeddings)
            c_embeddings = self.char_encoding(c_embeddings, lens=c_lens, mask=c_masks)
            embeddings = torch.cat([t_embeddings, c_embeddings], dim=-1)
        else:
            t_indexs, = self.preprocess_sentences(sentences)
            t_indexs = t_indexs.to(self.device)
            masks = self.masking(t_indexs, mask_val=0)
            embeddings = self.token_embedding(t_indexs)
        return embeddings, masks
    