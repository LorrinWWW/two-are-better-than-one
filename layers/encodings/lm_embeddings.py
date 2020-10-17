
import os
import math
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import flair
from flair.embeddings import BertEmbeddings
from flair.data import *

from transformers import *

from utils import *
from functions import *

from layers.indexings import *


def form_sentence(tokens):
    s = Sentence()
    for w in tokens:
        s.add_token(Token(w))
    return s

def get_embs(s):
    ret = []
    for t in s:
        ret.append(t.get_embedding().cpu().numpy())
    return np.stack(ret, axis=0)


class PreEmbeddedLM(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config 
        self.device = config.device
        self.lm_emb_path = config.lm_emb_path
        
        self.lm_emb_is_file = os.path.isfile(self.lm_emb_path)
        
        if self.lm_emb_is_file:
            with open(self.lm_emb_path, 'rb') as f:
                self.emb_dict = pickle.load(f)
        elif 'bert' in self.lm_emb_path:
            print(f'{self.lm_emb_path} is not file, try load as bert model.')
            self.lm = [BertEmbeddings(
                self.lm_emb_path, layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean")]
            
            
    def forward(self, batch_tokens):
        
        tmp = self.emb_tokens(batch_tokens)
        if isinstance(tmp[0], list) or isinstance(tmp[0], tuple):
            embs = [x[0] for x in tmp]
            heads = [x[1] for x in tmp]
        else:
            embs = tmp
            heads = None
            
        embs = pad_sequences(embs, maxlen=None, dtype='float32',
                  padding='post', truncating='post', value=0.)
        embs = torch.from_numpy(embs).float()
        embs = embs.to(self.device)
        
        if heads is not None:
            heads = pad_matrices(heads, maxlen=None, dtype='float32',
                  padding='post', truncating='post', value=0.)
            heads = torch.from_numpy(heads).float()
            
            heads = heads.to(self.device)
        
        return (embs, heads)
    
    def emb_tokens(self, tokens_list):
        
        rets = []
        
        if self.lm_emb_is_file:
        
            for tokens in tokens_list:
                tokens = tuple(tokens)
                if tokens not in self.emb_dict:
                    raise Exception(f'{tokens} not pre-emb')
                
                rets.append(self.emb_dict[tokens])
        
        else:
            
            s = [form_sentence(tuple(tokens)) for tokens in tokens_list]
            try:
                self.lm[0].embed(s)
            except Exception as e:
                for _s in s:
                    print(_s)
                raise e
            rets = [get_embs(_s) for _s in s]
            
        return rets
    

class BERTEmbedding(nn.Module):
    
    def __init__(self, ckpt_name='bert-base-uncased'):
        super().__init__()
        
#         self.config = config
#         self.device = config.device
        self.ckpt_name = ckpt_name
        self.model = BertModel.from_pretrained(ckpt_name)
        self.tokenizer = BertTokenizer.from_pretrained(ckpt_name)
        
    def preprocess_sentences(self, sentences):
        
        if len(sentences) > 1 and isinstance(sentences[1], torch.Tensor):
            # sentences is Tensor or is a list/tuple of Tensor
            return sentences #[(x.to(self.device) if isinstance(x, torch.Tensor) else x) for x in sentences]
        
        if 'uncased' in self.ckpt_name:
            sentences = [[w.lower() for w in s] for s in sentences]
        
        idxs = [self.tokenizer.convert_tokens_to_ids(s) for s in sentences]
        idxs = pad_sequences(
            idxs, maxlen=None, dtype='int64',
            padding='post', truncating='post', value=0.)
        
        idxs = torch.from_numpy(idxs)
        
        return [sentences, idxs]
        
    def forward(self, sentences):
        ret = self.model(sentences)
#         print(ret[0].shape, ret[1].shape)
        return ret[0]
    
    
class LMAllEmbedding(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.lm_embedding = BERTEmbedding(ckpt_name='bert-large-uncased')
        self.masking = Masking()
        
    def load_pretrained(self, path, freeze=True):
        pass
    
    def preprocess_sentences(self, sentences):
        return self.lm_embedding.preprocess_sentences(sentences)
    
    def forward(self, sentences):
        sentences, t_indexs = self.preprocess_sentences(sentences)
        t_indexs = t_indexs.to(self.config.device)
        masks = self.masking(t_indexs, mask_val=0)
        return self.lm_embedding(t_indexs), masks
        
        
        