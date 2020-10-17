
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from typing import *

from utils import *
from functions import *

import torch.jit as jit 

########## CELLS ##############
### 2d ####

class GRU2dCell(jit.ScriptModule):
    
    __constants__ = ['input_dim', 'state_dim']
    
    def __init__(self, input_dim, state_dim, dropout=0.):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        self.Wi = nn.Linear(self.input_dim, self.state_dim*4)
        self.Ws = nn.Linear(self.state_dim*2, self.state_dim*4)

    @jit.script_method
    def forward(self, x, s_prev0, s_prev1):
        
        #s_prev0 = torch.zeros_like(s_prev0)
        #s_prev1 = torch.zeros_like(s_prev1)
        
        s = torch.cat([s_prev0, s_prev1], -1)
        igates = self.Wi(x)
        sgates = self.Ws(s)
        gates = igates + sgates

        # r_inv actual represents (1-r)
        r_inv, i, n, l = gates.chunk(4, 1)
        s_n = sgates[:, self.state_dim*2:self.state_dim*3]
        
        l = l.sigmoid()
        r_inv = r_inv.sigmoid()
        i = i.sigmoid()
        n = (n - r_inv*s_n).tanh() # <==> (i_n + r * s_n)
        
        h = n + i * (l*s_prev0 + (1.-l)*s_prev1 - n)

        return h


class LNGRU2dCell(jit.ScriptModule):
    
    __constants__ = ['input_dim', 'state_dim']
    
    def __init__(self, input_dim, state_dim, dropout=0.):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        self.Wi = nn.Linear(self.input_dim, self.state_dim * 4, bias=None)
        self.Ws = nn.Linear(self.state_dim * 2, self.state_dim * 4, bias=None)
        self.LNi = nn.LayerNorm(self.state_dim * 4)
        self.LNs = nn.LayerNorm(self.state_dim * 4)
        self.LNh = nn.LayerNorm(self.state_dim)
        self.dropout_layer = nn.Dropout(dropout, inplace=True)

    @jit.script_method
    def forward(self, x, s_prev0, s_prev1):
        
        s = torch.cat([s_prev0, s_prev1], -1)
        igates = self.dropout_layer(self.LNi(self.Wi(x)))
        sgates = self.dropout_layer(self.LNs(self.Ws(s)))
        gates = igates + sgates

        # r_inv actual represents (1-r)
        r_inv, i, n, l = gates.chunk(4, 1)
        s_n = sgates[:, self.state_dim*2:self.state_dim*3]
        
        l = l.sigmoid()
        r_inv = r_inv.sigmoid()
        i = i.sigmoid()
        n = (n - r_inv*s_n).tanh() # <==> (i_n + r * s_n)
        
        h = n + i * (l*s_prev0 + (1.-l)*s_prev1 - n)
        
        h = self.dropout_layer(self.LNh(h))

        return h
    
### 3d ###
class GRU3dCell(jit.ScriptModule):
    
    __constants__ = ['input_dim', 'state_dim']
    
    def __init__(self, input_dim, state_dim, dropout=0.):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        self.Wi = nn.Linear(self.input_dim, self.state_dim * 6)
        self.Ws = nn.Linear(self.state_dim * 3, self.state_dim * 6)

    @jit.script_method
    def forward(self, x, s_prev0, s_prev1, s_prev2):
        
        #s_prev2 = torch.zeros_like(s_prev2)
        #s_prev0 = torch.zeros_like(s_prev0)
        #s_prev1 = torch.zeros_like(s_prev1)
        
        s = torch.cat([s_prev0, s_prev1, s_prev2], -1)
        igates = self.Wi(x)
        sgates = self.Ws(s)
        gates = igates + sgates

        # r_inv actual represents (1-r)
        r_i_n, l = gates.chunk(2, 1)
        r_inv, i, n = r_i_n.chunk(3, 1)
        s_n = sgates[:, self.state_dim*2:self.state_dim*3]
        
        l = l.view(-1, 3, self.state_dim).softmax(1) # weights for 3 hidden states
        r_inv = r_inv.sigmoid()
        i = i.sigmoid()
        n = (n - r_inv*s_n).tanh() # <==> (i_n + r * s_n)
        
        h = n + i * ( (l*s.view(-1,3,self.state_dim)).sum(1) - n)

        return h
    
    
class GRU25dCell(GRU3dCell):
    pass


class LNGRU3dCell(jit.ScriptModule):
    
    __constants__ = ['input_dim', 'state_dim']
    
    def __init__(self, input_dim, state_dim, dropout=0.):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        self.Wi = nn.Linear(self.input_dim, self.state_dim * 6, bias=None)
        self.Ws = nn.Linear(self.state_dim * 3, self.state_dim * 6, bias=None)
        self.LNi = nn.LayerNorm(self.state_dim * 6)
        self.LNs = nn.LayerNorm(self.state_dim * 6)
        self.LNh = nn.LayerNorm(self.state_dim)
        self.dropout_layer = nn.Dropout(dropout, inplace=True)

    @jit.script_method
    def forward(self, x, s_prev0, s_prev1, s_prev2):
        
        s = torch.cat([s_prev0, s_prev1, s_prev2], -1)
        igates = self.dropout_layer(self.LNi(self.Wi(x)))
        sgates = self.dropout_layer(self.LNs(self.Ws(s)))
        gates = igates + sgates

        # r_inv actual represents (1-r)
        r_i_n, l = gates.chunk(2, 1)
        r_inv, i, n = r_i_n.chunk(3, 1)
        s_n = sgates[:, self.state_dim*2:self.state_dim*3]
        
        l = l.view(-1, 3, self.state_dim).softmax(1) # weights for 3 hidden states
        r_inv = r_inv.sigmoid()
        i = i.sigmoid()
        n = (n - r_inv*s_n).tanh() # <==> (i_n + r * s_n)
        
        h = n + i * ( (l*s.view(-1,3,self.state_dim)).sum(1) - n)
        
        h = self.dropout_layer(self.LNh(h))

        return h
    

class LNGRU25dCell(LNGRU3dCell):
    pass
    
    
############# Layer ###############

class GRU2dLayer(jit.ScriptModule):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None, _Cell=LNGRU3dCell):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim
        
        self.cell = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        
    @jit.script_method
    def forward(self, x: torch.Tensor, states: Optional[torch.Tensor], masks: torch.Tensor):
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = x.flip(1)
        
        masks = masks.permute(1, 2, 0).unsqueeze(-1).float()
        masks = masks.flip(1)
        
        if states is None:
            states = torch.zeros(T0+1, T1+1, B, H, device=x.device) # (T0+1, T1+1*, B, H)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_prev0 = s_current[-diag_len:].view(new_batch_size, H)
            s_prev1 = s_current[:diag_len].view(new_batch_size, H)
            
            # run batched computation for this diagonal
            s_next = self.cell(
                x_current, s_prev0, s_prev1)
            
            # separate batch and diag_len again so we can store them accordingly
            to_save = s_next.view(diag_len, B, H)
            to_save = to_save * masks.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1].permute(2, 0, 1, 3) # B, T0 T1 H*2
        states_s = states_s.flip(2)
        
        return states_s, states # (B, T, H), and (T0+1, T1+1*, B, H)
    
class BGRU2dLayer(jit.ScriptModule):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None, _Cell=LNGRU2dCell):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim // 2
        
        self.cellf = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        self.cellb = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        
    @jit.script_method
    def forward(self, 
                x: torch.Tensor, 
                states: Optional[torch.Tensor],
                masks: torch.Tensor):
        
        assert states is None
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = torch.cat([x.flip(1), x.flip(0)], -1)
        
        masks = masks.permute(1, 2, 0).unsqueeze(-1).float().repeat(1, 1, 1, H) # (T0, T1, B, H)
        masks = torch.cat([masks.flip(1), masks.flip(0)], -1)
        
        states = torch.zeros(T0+1, T1+1, B, H*2, device=x.device) # (T0+1, T1+1*, B, H)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E*2)
            x_current_f, x_current_b = x_current.chunk(2, -1)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_prev_f0, s_prev_b0 = s_current[-diag_len:].view(new_batch_size, H*2).chunk(2, 1)
            s_prev_f1, s_prev_b1 = s_current[:diag_len].view(new_batch_size, H*2).chunk(2, 1)
            
            # run batched computation for this diagonal
            s_next_f = self.cellf(
                x_current_f, s_prev_f0, s_prev_f1)
            
            s_next_b = self.cellb(
                x_current_b, s_prev_b0, s_prev_b1)
            
            # separate batch and diag_len again so we can store them accordingly
            to_save = torch.cat([s_next_f, s_next_b], -1).view(diag_len, B, H*2)
            to_save = to_save * masks.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1].permute(2, 0, 1, 3) # B, T0 T1 H*2
        tmp0, tmp1 = states_s.chunk(2, -1)
        states_s = torch.cat([tmp0.flip(2), tmp1.flip(1)], -1)
        
        return states_s, states # (B, T, H), and (T0+1, T1+1*, B, H)
    
    
class BsGRU2dLayer(jit.ScriptModule):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None, _Cell=LNGRU2dCell):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim // 2
        
        self.cellf = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        self.cellb = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        
    @jit.script_method
    def forward(self, 
                x: torch.Tensor, 
                states: Optional[torch.Tensor],
                masks: torch.Tensor):
        
        assert states is None
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = torch.cat([x, x.flip(0).flip(1)], -1)
        
        masks = masks.permute(1, 2, 0).unsqueeze(-1).float().repeat(1, 1, 1, H) # (T0, T1, B, H)
        masks = torch.cat([masks, masks.flip(0).flip(1)], -1)
        
        states = torch.zeros(T0+1, T1+1, B, H*2, device=x.device) # (T0+1, T1+1*, B, H)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E*2)
            x_current_f, x_current_b = x_current.chunk(2, -1)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_prev_f0, s_prev_b0 = s_current[-diag_len:].view(new_batch_size, H*2).chunk(2, 1)
            s_prev_f1, s_prev_b1 = s_current[:diag_len].view(new_batch_size, H*2).chunk(2, 1)
            
            # run batched computation for this diagonal
            s_next_f = self.cellf(
                x_current_f, s_prev_f0, s_prev_f1)
            
            s_next_b = self.cellb(
                x_current_b, s_prev_b0, s_prev_b1)
            
            # separate batch and diag_len again so we can store them accordingly
            to_save = torch.cat([s_next_f, s_next_b], -1).view(diag_len, B, H*2)
            to_save = to_save * masks.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1].permute(2, 0, 1, 3) # B, T0 T1 H*2
        tmp0, tmp1 = states_s.chunk(2, -1)
        states_s = torch.cat([tmp0, tmp1.flip(1).flip(2)], -1)
        
        return states_s, states # (B, T, H), and (T0+1, T1+1*, B, H)
    

class QGRU2dLayer(jit.ScriptModule):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None, _Cell=LNGRU2dCell):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim // 4
        
        self.cella = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        self.cellb = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        self.cellc = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        self.celld = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        
    @jit.script_method
    def forward(self, 
                x: torch.Tensor, 
                states: Optional[torch.Tensor],
                masks: torch.Tensor):
        
        assert states is None
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = torch.cat([x.flip(1), x.flip(0), x, x.flip(1).flip(0)], -1)
        
        masks = masks.permute(1, 2, 0).unsqueeze(-1).float().repeat(1, 1, 1, H) # (T0, T1, B, H)
        masks = torch.cat([masks.flip(1), masks.flip(0), masks, masks.flip(1).flip(0)], -1)
        
        states = torch.zeros(T0+1, T1+1, B, H*4, device=x.device) # (T0+1, T1+1*, B, H)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E*4)
            x_current_a, x_current_b, x_current_c, x_current_d = x_current.chunk(4, -1)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_prev_a0, s_prev_b0, s_prev_c0, s_prev_d0 = s_current[-diag_len:].view(new_batch_size, H*4).chunk(4, 1)
            s_prev_a1, s_prev_b1, s_prev_c1, s_prev_d1 = s_current[:diag_len].view(new_batch_size, H*4).chunk(4, 1)
            
            # run batched computation for this diagonal
            s_next_a = self.cella(
                x_current_a, s_prev_a0, s_prev_a1)
            
            s_next_b = self.cellb(
                x_current_b, s_prev_b0, s_prev_b1)
            
            s_next_c = self.cellc(
                x_current_c, s_prev_c0, s_prev_c1)
            
            s_next_d = self.celld(
                x_current_d, s_prev_d0, s_prev_d1)
            
            # separate batch and diag_len again so we can store them accordingly
            to_save = torch.cat([s_next_a, s_next_b, s_next_c, s_next_d], -1).view(diag_len, B, H*4)
            to_save = to_save * masks.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1].permute(2, 0, 1, 3) # B, T0 T1 H*4
        tmp0, tmp1, tmp2, tmp3 = states_s.chunk(4, -1)
        states_s = torch.cat([tmp0.flip(2), tmp1.flip(1), tmp2, tmp3.flip(1).flip(2)], -1)
        
        return states_s, states # (B, T, H), and (T0+1, T1+1*, B, H)
    
    
class GRU25dLayer(jit.ScriptModule):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None, _Cell=LNGRU3dCell):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim
        
        self.cell = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        
    @jit.script_method
    def forward(self, x: torch.Tensor, states: Optional[torch.Tensor], masks: torch.Tensor):
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = x.flip(1)
        
        masks = masks.permute(1, 2, 0).unsqueeze(-1).float()
        masks = masks.flip(1)
        
        if states is None:
            states_in = torch.zeros(T0+1, T1+1, B, H, device=x.device) # (T0+1, T1+1*, B, H)
            states = states_in.clone()
        else:
            states_in = states
            states = torch.zeros(T0+1, T1+1, B, H, device=x.device)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_next = states_in.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_prev0 = s_current[-diag_len:].view(new_batch_size, H)
            s_prev1 = s_current[:diag_len].view(new_batch_size, H)
            s_prev2 = s_next[-diag_len-1:diag_len+1].view(new_batch_size, H)
            
            # run batched computation for this diagonal
            s_next = self.cell(
                x_current, s_prev0, s_prev1, s_prev2)
            
            # separate batch and diag_len again so we can store them accordingly
            to_save = s_next.view(diag_len, B, H)
            to_save = to_save * masks.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1].permute(2, 0, 1, 3) # B, T0 T1 H*2
        states_s = states_s.flip(2)
        
        return states_s, states # (B, T, H), and (T0+1, T1+1*, B, H)
    


class BGRU25dLayer(jit.ScriptModule):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None, _Cell=LNGRU3dCell):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim // 2
        
        self.cellf = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        self.cellb = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        
    @jit.script_method
    def forward(self, 
                x: torch.Tensor, 
                states: Optional[torch.Tensor],
                masks: torch.Tensor):
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = torch.cat([x.flip(1), x.flip(0)], -1) # (T0, T1, B, E*2)
        
        masks = masks.permute(1, 2, 0).unsqueeze(-1).float().repeat(1, 1, 1, H) # (T0, T1, B, H)
        masks = torch.cat([masks.flip(1), masks.flip(0)], -1) # (T0, T1, B, H*2)
        
        if states is None:
            states_in = torch.zeros(T0+1, T1+1, B, H*2, device=x.device) # (T0+1, T1+1*, B, H)
            states = states_in.clone()
        else:
            states_in = states
            states = torch.zeros(T0+1, T1+1, B, H*2, device=x.device)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E*2)
            x_current_f, x_current_b = x_current.chunk(2, -1)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_next = states_in.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_prev_f0, s_prev_b0 = s_current[-diag_len:].view(new_batch_size, H*2).chunk(2, 1)
            s_prev_f1, s_prev_b1 = s_current[:diag_len].view(new_batch_size, H*2).chunk(2, 1)
            s_prev_f2, s_prev_b2 = s_next[-diag_len-1:diag_len+1].view(new_batch_size, H*2).chunk(2, 1)
            
            # run batched computation for this diagonal
            s_next_f = self.cellf(
                x_current_f, s_prev_f0, s_prev_f1, s_prev_f2)
            
            s_next_b = self.cellb(
                x_current_b, s_prev_b0, s_prev_b1, s_prev_b2)
            
            # separate batch and diag_len again so we can store them accordingly
            to_save = torch.cat([s_next_f, s_next_b], -1).view(diag_len, B, H*2)
            to_save = to_save * masks.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1].permute(2, 0, 1, 3) # B, T0 T1 H*2
        tmp0, tmp1 = states_s.chunk(2, -1)
        states_s = torch.cat([tmp0.flip(2), tmp1.flip(1)], -1)
        
        return states_s, states # (B, T, H), and (T0+1, T1+1*, B, H)
    

class BsGRU25dLayer(jit.ScriptModule):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None, _Cell=LNGRU3dCell):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim // 2
        
        self.cellf = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        self.cellb = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        
    @jit.script_method
    def forward(self, 
                x: torch.Tensor, 
                states: Optional[torch.Tensor],
                masks: torch.Tensor):
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = torch.cat([x, x.flip(0).flip(1)], -1) # (T0, T1, B, E*2)
        
        masks = masks.permute(1, 2, 0).unsqueeze(-1).float().repeat(1, 1, 1, H) # (T0, T1, B, H)
        masks = torch.cat([masks, masks.flip(0).flip(1)], -1) # (T0, T1, B, H*2)
        
        if states is None:
            states_in = torch.zeros(T0+1, T1+1, B, H*2, device=x.device) # (T0+1, T1+1*, B, H)
            states = states_in.clone()
        else:
            states_in = states
            states = torch.zeros(T0+1, T1+1, B, H*2, device=x.device)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E*2)
            x_current_f, x_current_b = x_current.chunk(2, -1)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_next = states_in.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_prev_f0, s_prev_b0 = s_current[-diag_len:].view(new_batch_size, H*2).chunk(2, 1)
            s_prev_f1, s_prev_b1 = s_current[:diag_len].view(new_batch_size, H*2).chunk(2, 1)
            s_prev_f2, s_prev_b2 = s_next[-diag_len-1:diag_len+1].view(new_batch_size, H*2).chunk(2, 1)
            
            # run batched computation for this diagonal
            s_next_f = self.cellf(
                x_current_f, s_prev_f0, s_prev_f1, s_prev_f2)
            
            s_next_b = self.cellb(
                x_current_b, s_prev_b0, s_prev_b1, s_prev_b2)
            
            # separate batch and diag_len again so we can store them accordingly
            to_save = torch.cat([s_next_f, s_next_b], -1).view(diag_len, B, H*2)
            to_save = to_save * masks.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1].permute(2, 0, 1, 3) # B, T0 T1 H*2
        tmp0, tmp1 = states_s.chunk(2, -1)
        states_s = torch.cat([tmp0, tmp1.flip(1).flip(2)], -1)
        
        return states_s, states # (B, T, H), and (T0+1, T1+1*, B, H)
    

class QGRU25dLayer(jit.ScriptModule):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None, _Cell=LNGRU3dCell):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim // 4
        
        self.cella = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        self.cellb = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        self.cellc = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        self.celld = _Cell(self.emb_dim, self.hidden_dim, dropout=0.0)
        
    @jit.script_method
    def forward(self, 
                x: torch.Tensor, 
                states: Optional[torch.Tensor],
                masks: torch.Tensor):
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = torch.cat([x.flip(1), x.flip(0), x, x.flip(0).flip(1)], -1) # (T0, T1, B, E*4)
        
        masks = masks.permute(1, 2, 0).unsqueeze(-1).float().repeat(1, 1, 1, H) # (T0, T1, B, H)
        masks = torch.cat([masks.flip(1), masks.flip(0), masks, masks.flip(0).flip(1)], -1) # (T0, T1, B, H*4)
        
        if states is None:
            states_in = torch.zeros(T0+1, T1+1, B, H*4, device=x.device) # (T0+1, T1+1*, B, H)
            states = states_in.clone()
        else:
            states_in = states
            states = torch.zeros(T0+1, T1+1, B, H*4, device=x.device)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E*4)
            x_current_a, x_current_b, x_current_c, x_current_d = x_current.chunk(4, -1)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_next = states_in.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_prev_a0, s_prev_b0, s_prev_c0, s_prev_d0 = s_current[-diag_len:].view(new_batch_size, H*4).chunk(4, 1)
            s_prev_a1, s_prev_b1, s_prev_c1, s_prev_d1 = s_current[:diag_len].view(new_batch_size, H*4).chunk(4, 1)
            s_prev_a2, s_prev_b2, s_prev_c2, s_prev_d2 = s_next[-diag_len-1:diag_len+1].view(new_batch_size, H*4).chunk(4, 1)
            
            # run batched computation for this diagonal
            s_next_a = self.cella(
                x_current_a, s_prev_a0, s_prev_a1, s_prev_a2)
            
            s_next_b = self.cellb(
                x_current_b, s_prev_b0, s_prev_b1, s_prev_b2)
            
            s_next_c = self.cellc(
                x_current_c, s_prev_c0, s_prev_c1, s_prev_c2)
            
            s_next_d = self.celld(
                x_current_d, s_prev_d0, s_prev_d1, s_prev_d2)
            
            # separate batch and diag_len again so we can store them accordingly
            to_save = torch.cat([s_next_a, s_next_b, s_next_c, s_next_d], -1).view(diag_len, B, H*4)
            to_save = to_save * masks.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1].permute(2, 0, 1, 3) # B, T0 T1 H*4
        tmp0, tmp1, tmp2, tmp3 = states_s.chunk(4, -1)
        states_s = torch.cat([tmp0.flip(2), tmp1.flip(1), tmp2, tmp3.flip(1).flip(2)], -1)
        #states_s = torch.cat([tmp0.flip(2), tmp1.flip(1), tmp0.flip(2), tmp1.flip(1)], -1)
        
        return states_s, states # (B, T, H), and (T0+1, T1+1*, B, H)
    
    
    
    
    