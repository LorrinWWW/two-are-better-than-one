
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import torch.jit as jit 

from utils import *
from functions import *

from layers.indexings import *
from layers.encodings import *


class MixReduce(jit.ScriptModule):
    
    __constants__ = ['n_in', 'n_out']
    
    def __init__(self, n_in, n_out):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        
        self.biaffine = Biaffine(n_in, n_out)
        
        self.reduce = nn.Linear(n_in*3, n_out)
        
    @jit.script_method
    def forward(self, x, y):
        
        z = self.biaffine(x, y)
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None])
        
        s = torch.cat([x, y, z], -1)
        
        s = self.reduce(s)
        
        return s


class CatReduce(jit.ScriptModule):
    
    __constants__ = ['n_in', 'n_out']
    
    def __init__(self, n_in, n_out):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        
        self.reduce = nn.Linear(n_in*2, n_out)
        
    @jit.script_method
    def forward(self, x, y):
        
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None])
        
        s = torch.cat([x, y], -1)
        
        s = self.reduce(s)
        
        return s

        
class Biaffine(jit.ScriptModule):
               
    __constants__ = ['n_in', 'n_out', 'bias_x', 'bias_y']

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.randn( n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        nn.init.xavier_uniform_(self.weight.data)

    @jit.script_method
    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->bxyo', x, self.weight, y)
        # remove dim 1 if n_out == 1
        #s = s.squeeze(1).permute(0, 2, 3, 1)

        return s