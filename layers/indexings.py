import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from itertools import combinations

from utils import *
from functions import *


def get_tag_indexing(config):
        
    if config.tag_form == 'iob2':
        _Indexing = BIOIndexing
    elif config.tag_form == 'iobes':
        _Indexing = BIOESIndexing
    else:
        _Indexing = Indexing
    indexing = _Indexing(
        maxlen=config.maxlen, vocab_file=config.vocab_file, cased=True)
    
    return indexing

class Indexing:
    def __init__(self, maxlen=None, vocab_file=None, cased=True, depth=2):
        
        self.depth = depth
        self.cased = cased
#         self.try_uncased = try_uncased
        self.maxlen = maxlen
        self.vocab_file = vocab_file
        
        if vocab_file is None:
            self.update_vocab = True
            self.vocab = {'[PAD]': 0, '[UNK]':1}
            self.inv_vocab = {}
        else:
            self.update_vocab = False
            self.vocab = {}
            self.inv_vocab = {}
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = {token.strip():i for i, token in enumerate(f)}
                
    def update_inv_vocab(self):
        if len(self.vocab) != len(self.inv_vocab):
            self.inv_vocab = {v:k for k,v in self.vocab.items()}

    def idx2token(self, idx):
        self.update_inv_vocab()
        return self.inv_vocab.get(idx, 'O')
                
    def token2idx(self, token):
        #self.update_inv_vocab()
        if not self.cased:
            token = token.lower()
        if token in self.vocab:
            return self.vocab[token]
#         elif self.cased and self.try_uncased and token.lower() in self.vocab:
#             return self.vocab[token.lower()]
        elif self.update_vocab:
            self.vocab[token] = len(self.vocab)
            return self.vocab[token]
        else:
#             print(token)
            return self.vocab['[UNK]']
        
    def idxs2tokens(self, idxs):
        if isinstance(idxs, str):
            return idxs
        elif isinstance(idxs, int) or (type(idxs).__module__=='numpy' and idxs.shape==()):
            return self.idx2token(idxs)
        elif hasattr(idxs, '__iter__'):
            return [self.idxs2tokens(_idxs) for _idxs in idxs]
        else:
            raise Exception(f'Unkown type: ({type(idxs)}){idxs}')

    def tokens2idxs(self, tokens, depth=None):
        '''
        tokens: arbitrary depth iterable object, ending with str object.
        '''
        if depth is None:
            depth = self.depth
        if isinstance(tokens, torch.Tensor):
            return tokens
        elif depth > 0:
            return [self.tokens2idxs(_tokens, depth=depth-1) for _tokens in tokens]
        return self.token2idx(tokens)

    def __call__(self, tokens, add_begin=-1, add_end=-1):
        indexs = self.tokens2idxs(tokens)
        if add_begin >= 0:
            indexs = [[add_begin]+_idxs for _idxs in indexs]
        if add_end >= 0:
            indexs = [_idxs+[add_end] for _idxs in indexs]
        if self.depth >= 2:
            maxlen = self.maxlen if self.maxlen else max(len(_idxs) for _idxs in indexs)
            indexs = pad_sequences(indexs, maxlen=self.maxlen, dtype='long',
                                   padding='post', truncating='post', value=0) # 0 == self.token_indexing.vocab['[PAD]']
        else:
            indexs = np.array(indexs, dtype=np.int64)
        indexs = torch.from_numpy(indexs)
        return indexs
    
    def inv(self, idxs):
        return self.idxs2tokens(idxs)
    

class IndexingMatrix(Indexing):
    def __call__(self, tokens):
        indexs = self.tokens2idxs(tokens)
        if self.depth >= 2:
            maxlen = self.maxlen if self.maxlen else max(len(_idxs) for _idxs in indexs)
            indexs = pad_matrices(indexs, maxlen=self.maxlen, dtype='long',
                           padding='post', truncating='post', value=0) # 0 == self.token_indexing.vocab['[PAD]']
        else:
            indexs = np.array(indexs, dtype=np.int64)
        indexs = torch.from_numpy(indexs)
        return indexs
    
class BIOIndexing(Indexing):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.vocab = {'O': 0,}
        # O is 0
        # B- is 2k+1
        # I- is 2k+2, k>=0
        # e.g. O, B-A, I-A, B-B, I-B
        #      0,   1,   2,   3,   4
    
    def token2idx(self, token):
        if not self.cased:
            token = token.lower()
        if token in self.vocab:
            return self.vocab[token]
        elif self.update_vocab:
            # add B-, I-, if not in vocab
            if token[0] not in ['B', 'I']:
                print(f'unknown tag: {token}')
                token = 'B-'+token
                print(f'auto-convert to: {token}')
            self.vocab['B'+token[1:]] = len(self.vocab)
            self.vocab['I'+token[1:]] = len(self.vocab)
            return self.vocab[token]
        else:
            return self.vocab['[UNK]']
        
class BIOESIndexing(Indexing):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.vocab = {'O': 0,}
        # O is 0
        # B- is 4k+1
        # I- is 4k+2, k>=0
        # E- is 4k+3, 
        # S- is 4k+4
    
    def token2idx(self, token):
        if not self.cased:
            token = token.lower()
        if token in self.vocab:
            return self.vocab[token]
        elif self.update_vocab:
            if token[0] not in ['B', 'I', 'E', 'S']:
                print(f'unknown tag: {token}')
                token = 'B-'+token
                print(f'auto-convert to: {token}')
            self.vocab['B'+token[1:]] = len(self.vocab)
            self.vocab['I'+token[1:]] = len(self.vocab)
            self.vocab['E'+token[1:]] = len(self.vocab)
            self.vocab['S'+token[1:]] = len(self.vocab)
            return self.vocab[token]
        else:
            return self.vocab['[UNK]']

    
class CharIndexing(Indexing):
    def tokens2idxs(self, tokens, depth=3):
        '''
        tokens: arbitrary depth iterable object, ending with str object.
        '''
        if depth > 0:
            return [self.tokens2idxs(_tokens, depth=depth-1) for _tokens in tokens]
        return self.token2idx(tokens)

    def __call__(self, tokens, add_begin=-1, add_end=-1, pad_position='post'):
        indexs = self.tokens2idxs(tokens) # (bs, tokens, chars)
        if add_begin >= 0:
            indexs = [[[add_begin]]+_idxs for _idxs in indexs]
        if add_end >= 0:
            indexs = [_idxs+[[add_end]] for _idxs in indexs]
        t_maxlen = self.maxlen if self.maxlen else max(len(_idxs) for _idxs in indexs) # maxlen of tokens
        c_maxlen = max(max(len(__idxs) for __idxs in _idxs) for _idxs in indexs) # maxlen of chars
        
        pad_token = [0]*c_maxlen
        
        for i, tokens in enumerate(indexs):
            for j, chars in enumerate(tokens):
                if pad_position == 'post':
                    tokens[j] = chars + [0]*(c_maxlen - len(chars))
                elif pad_position == 'both':
                    n_left = (c_maxlen - len(chars)) // 2
                    n_right = (c_maxlen - len(chars)) - n_left
                    tokens[j] = [0]*n_left + chars + [0]*n_right
                else:
                    raise Exception(f"no such position")
            indexs[i] = tokens + [pad_token]*(t_maxlen - len(tokens))
        indexs = np.array(indexs, dtype='long')
        indexs = torch.from_numpy(indexs)
        return indexs
        
class Masking:
    def __call__(self, idxs, mask_val=0, dtype=None):
        if dtype is None or dtype == torch.BoolTensor:
            masks = (idxs!=mask_val).to(idxs.device)
        else:
            masks = (idxs!=mask_val).type(dtype).to(idxs.device)
        return masks
