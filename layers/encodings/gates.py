

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


class SigmoidGate(nn.Module):
    
    def __init__(self, hidden_dim, out_dim=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else hidden_dim
        
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Sigmoid(),
        )
        init_linear(self.gate[0])
        
    def forward(self, q, c, only_gate=False):
        '''
        q (XXX T, H)
        c (XXX H)
        '''
        g = self.gate(c.unsqueeze(-2))
        
        if only_gate:
            return g
        return q * g, g
    
    
class TanhGate(nn.Module):
    
    def __init__(self, hidden_dim, out_dim=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else hidden_dim
        
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Tanh(),
        )
        init_linear(self.gate[0])
        
    def forward(self, q, c, only_gate=False):
        '''
        q (XXX T, H)
        c (XXX H)
        '''
        g = self.gate(c.unsqueeze(-2))
        
        if only_gate:
            return g
        return q * g, g
    
    
class SlotGate(nn.Module):
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.Wc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Vg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        init_linear(self.Wc)
        init_linear(self.Vg)
        
    def forward(self, q, c, only_gate=False):
        '''
        q (XXX T, H)
        c (XXX H)
        '''
        g = torch.tanh(q + self.Wc(c).unsqueeze(-2)) # BTH * B1H = BTH
        g = self.Vg(g) # BTH
        g = torch.sum(g, dim=-2, keepdims=True) # B1H
        
        if only_gate:
            return g
        return q * g
    

        
        