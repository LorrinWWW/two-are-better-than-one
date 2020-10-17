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
    

class LSTM2dCell(jit.ScriptModule):
    """
    A 2d-LSTM Cell that computes it's hidden state and cell state based on
        - an input x
        - the previous horizontal hidden and cell state
        - the previous vertical hidden and cell state
    Args:
        input_dim: the input dimension (i.e. second dimension of x)
        state_dim: dimension of the hidden and cell state of this LSTM unit
        device: the device (CPU / GPU) to run all computations on / store tensors on
    """
    
    __constants__ = ['input_dim', 'state_dim']
    
    def __init__(self, input_dim, state_dim, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device
        
        self.W = nn.Linear(self.input_dim + self.state_dim * 2, self.state_dim * 5).to(self.device)
        
        # init weight
        #sampling_range = np.sqrt(6.0 / (self.state_dim + self.input_dim + self.state_dim * 2))
        #nn.init.uniform_(self.W.weight.data, -sampling_range, sampling_range)
        #nn.init.orthogonal_(self.W.weight.data)
        # init bias
        #self.W.bias.data.zero_()
        #self.W.bias.data[self.state_dim:self.state_dim*2].fill_(1.) # forget gate to be 1.
        

    @jit.script_method
    def forward(self, x, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver):
        """
        Forward pass of the 2d-LSTM Cell at horizontal step j and vertical step i (to compute c_ji and s_ji)
        Args:
            x: (batch x input_dim) input at horizontal step j
            s_prev_hor: (batch x state_dim) hidden state of cell at previous horizontal step j-1, same vertical step i
            s_prev_ver: (batch x state_dim) hidden state of cell at previous vertical step i-1, same horizontal step j
            c_prev_hor: (batch x state_dim) cell state of cell at previous horizontal step j-1, same vertical step i
            c_prev_ver: (batch x state_dim) cell state of cell at previous vertical step i-1, same horizontal step j
        Returns:
            c: (batch x state_dim) next cell state (c_ji)
            s: (batch x state_dim) next hidden state (s_ji)
        """
        pre_activation = self.W(torch.cat([x, s_prev_hor, s_prev_ver], -1))

        # retrieve input, forget, output and lambda gate from gates
        i, f, o, l, c = pre_activation.chunk(5, 1)
        i = i.sigmoid()
        f = f.sigmoid()
        o = o.sigmoid()
        l = l.sigmoid()
        c = c.tanh()
        
        c = f * (l * c_prev_hor + (1 - l) * c_prev_ver) + c * i 
        s = c.tanh() * o

        return c, s
        
        
class LSTMEncoding2d(jit.ScriptModule):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim
        
        self.cell2d = LSTM2dCell(self.emb_dim, self.hidden_dim)
    
    @jit.script_method
    def forward(self, x):
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = x.flip(1) # (T0, T1*, B, E)
        
        states = torch.zeros(T0+1, T1+1, B, H*2, device=x.device) # (T0+1, T1+1*, B, E)
        
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_prev_hor, c_prev_hor = s_current[-diag_len:].view(new_batch_size, H*2).chunk(2, 1)
            s_prev_ver, c_prev_ver = s_current[:diag_len].view(new_batch_size, H*2).chunk(2, 1)
            
            # run batched computation for this diagonal
            c_next, s_next = self.cell2d(x_current, s_prev_hor, s_prev_ver, c_prev_hor, c_prev_ver)
            
            # separate batch and diag_len again so we can store them accordingly
            to_save = torch.cat([s_next, c_next], -1).view(diag_len, B, H*2)
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_next[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:,:-1,:,:H]
        states_s = states_s.permute(2, 0, 1, 3) # B, T0 T1 H
        states_s = states_s.flip(2)
        
        return states_s
    
class LSTMEncoding2d2way(jit.ScriptModule):
    '''
    left -> right, up -> down && left <- right, up <- down
    '''
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim // 2
        
        self.cell2df = LSTM2dCell(self.emb_dim, self.hidden_dim)
        self.cell2db = LSTM2dCell(self.emb_dim, self.hidden_dim)
        
    @jit.script_method
    def forward(self, x):
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        xf = x.flip(1)
        xb = x.flip(0)
        
        states = torch.zeros(T0+1, T1+1, B, H*4, device=x.device) # (T0, T1+1*, B, E*4)
        #  pad pad pad
        #   e   e  pad
        #   e   e  pad
        
        for offset in range(T1-1, -T0, -1):
            
            x_current_f = xf.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            x_current_b = xb.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current_f.size(0)
            new_batch_size = diag_len * B
            
            x_current_f = x_current_f.view(new_batch_size, E)
            x_current_b = x_current_b.view(new_batch_size, E)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = states.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            s_next = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_prev_hor_f, s_prev_hor_b, c_prev_hor_f, c_prev_hor_b = s_current[-diag_len:].view(new_batch_size, H*4).chunk(4, 1)
            s_prev_ver_f, s_prev_ver_b, c_prev_ver_f, c_prev_ver_b = s_current[:diag_len].view(new_batch_size, H*4).chunk(4, 1)
            
            # run batched computation for this diagonal
            c_next_f, s_next_f = self.cell2df(x_current_f, s_prev_hor_f, s_prev_ver_f, c_prev_hor_f, c_prev_ver_f)
            c_next_b, s_next_b = self.cell2db(x_current_b, s_prev_hor_b, s_prev_ver_b, c_prev_hor_b, c_prev_ver_b)
            
            # separate batch and diag_len again so we can store them accordingly
            tmp = torch.cat([s_next_f, s_next_b, c_next_f, c_next_b], -1).view(diag_len, B, H*4)
            s_next[-diag_len-1:diag_len+1] = tmp
            
        states_s = states[1:, :-1, :, :H*2].permute(2, 0, 1, 3) # B, T0 T1 H*4
        tmp0, tmp1 = states_s.chunk(2, -1)
        states_s = torch.cat([tmp0.flip(2), tmp1.flip(1)], -1)
        
        return states_s
    
    
################
    
class LSTM2dCellPlus(jit.ScriptModule):
    """
    A 2d-LSTM Cell that computes it's hidden state and cell state based on
        - an input x
        - the previous horizontal hidden and cell state
        - the previous vertical hidden and cell state
    Args:
        input_dim: the input dimension (i.e. second dimension of x)
        state_dim: dimension of the hidden and cell state of this LSTM unit
        device: the device (CPU / GPU) to run all computations on / store tensors on
    """
    
    __constants__ = ['input_dim', 'state_dim']
    
    def __init__(self, input_dim, state_dim, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device

        self.W = nn.Linear(self.input_dim + self.state_dim * 3, self.state_dim * 7).to(self.device)
        
        # init weight
        #sampling_range = np.sqrt(6.0 / (self.state_dim + self.input_dim + self.state_dim * 3))
        #nn.init.uniform_(self.W.weight.data, -sampling_range, sampling_range)
        ## init bias
        #self.W.bias.data.zero_()
        #self.W.bias.data[self.state_dim:self.state_dim*2].fill_(1.) # forget gate to be 1.

    @jit.script_method
    def forward(self, x, s_prev_hor, s_prev_ver, s_prev_dia, c_prev_hor, c_prev_ver, c_prev_dia):
        
        pre_activation = self.W(torch.cat([x, s_prev_hor, s_prev_ver, s_prev_dia], -1))

        # retrieve input, forget, output and lambda gate from gates
        i, f, o, l0, l1, l2, c = pre_activation.chunk(7, 1)
        l = torch.stack([l0,l1,l2], 1).softmax(1)
        i = i.sigmoid()
        f = f.sigmoid()
        o = o.sigmoid()
        c = c.tanh()
        
        c = f * (l[:,0] * c_prev_hor + l[:,1] * c_prev_ver + l[:,2] * c_prev_dia) + c * i
        s = c.tanh() * o

        return c, s
    
    
class LSTMEncoding2dPlus(jit.ScriptModule):
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim
        
        self.cell2d = LSTM2dCellPlus(self.emb_dim, self.hidden_dim)
    
    @jit.script_method
    def forward(self, x):
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        x = x.flip(1) # (T0, T1*, B, E)
        
        states = torch.zeros(T0+1, T1+1, B, H*2, device=x.device) # (T0+1, T1+1*, B, E)
        
        s_last = states.diagonal(offset=T1, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
        s_current = states.diagonal(offset=T1-1, dim1=0, dim2=1).permute(-1, 0, 1)
        for offset in range(T1-1, -T0, -1):
            
            x_current = x.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current_f.size(0)
            new_batch_size = diag_len * B
            
            x_current = x_current.view(new_batch_size, E)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = s_current.contiguous()
            s_prev_dia, c_prev_dia = s_last[-diag_len-1:diag_len+1].view(new_batch_size, H*2).chunk(2, 1)
            s_prev_hor, c_prev_hor = s_current[-diag_len:].view(new_batch_size, H*2).chunk(2, 1)
            s_prev_ver, c_prev_ver = s_current[:diag_len].view(new_batch_size, H*2).chunk(2, 1)
            
            # run batched computation for this diagonal
            c_next, s_next = self.cell2d(x_current, 
                                         s_prev_hor, s_prev_ver, s_prev_dia, 
                                         c_prev_hor, c_prev_ver, c_prev_dia,)
            
            # store them
            s_last = s_current
            to_save = torch.cat([s_next, c_next], -1).view(diag_len, B, H*4)
            s_current = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_current[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:,:-1,:,:H]
        states_s = states_s.permute(2, 0, 1, 3) # B, T0 T1 H
        states_s = states_s.flip(2)
        
        return states_s
    
class LSTMEncoding2d2wayPlus(jit.ScriptModule):
    '''
    left -> right, up -> down && left <- right, up <- down
    '''
    
    __constants__ = ['emb_dim', 'hidden_dim']
    
    def __init__(self, config, emb_dim=None):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim // 2
        
        self.cell2df = LSTM2dCellPlus(self.emb_dim, self.hidden_dim)
        self.cell2db = LSTM2dCellPlus(self.emb_dim, self.hidden_dim)
        
    @jit.script_method
    def forward(self, x):
        
        # x (B, T0, T1, H)
        B, T0, T1, E = x.shape
        H = self.hidden_dim
        
        x = x.permute(1, 2, 0, 3) # (T0, T1, B, E)
        xf = x.flip(1)
        xb = x.flip(0)
        
        states = torch.zeros(T0+1, T1+1, B, H*4, device=x.device) # (T0, T1+1*, B, E*4)
        #  pad pad pad
        #   e   e  pad
        #   e   e  pad
        
        s_last = states.diagonal(offset=T1, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
        s_current = states.diagonal(offset=T1-1, dim1=0, dim2=1).permute(-1, 0, 1)
        for offset in range(T1-1, -T0, -1):
            
            x_current_f = xf.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            x_current_b = xb.diagonal(offset=offset, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
            
            diag_len = x_current_f.size(0)
            new_batch_size = diag_len * B
            
            x_current_f = x_current_f.view(new_batch_size, E)
            x_current_b = x_current_b.view(new_batch_size, E)
            
            # calculate previous hidden & cell states for this diagonal
            s_current = s_current.contiguous()
            s_prev_dia_f, s_prev_dia_b, c_prev_dia_f, c_prev_dia_b = s_last[-diag_len-1:diag_len+1].view(new_batch_size, H*4).chunk(4, 1)
            s_prev_hor_f, s_prev_hor_b, c_prev_hor_f, c_prev_hor_b = s_current[-diag_len:].view(new_batch_size, H*4).chunk(4, 1)
            s_prev_ver_f, s_prev_ver_b, c_prev_ver_f, c_prev_ver_b = s_current[:diag_len].view(new_batch_size, H*4).chunk(4, 1)
            
            # run batched computation for this diagonal
            c_next_f, s_next_f = self.cell2df(x_current_f, 
                                              s_prev_hor_f, s_prev_ver_f, s_prev_dia_f, 
                                              c_prev_hor_f, c_prev_ver_f, c_prev_dia_f)
            c_next_b, s_next_b = self.cell2db(x_current_b, 
                                              s_prev_hor_b, s_prev_ver_b, s_prev_dia_b,
                                              c_prev_hor_b, c_prev_ver_b, c_prev_dia_b)
            
            # separate batch and diag_len again so we can store them accordingly
            s_last = s_current
            to_save = torch.cat([s_next_f, s_next_b, c_next_f, c_next_b], -1).view(diag_len, B, H*4)
            s_current = states.diagonal(offset=offset-1, dim1=0, dim2=1).permute(-1, 0, 1)
            s_current[-diag_len-1:diag_len+1] = to_save
            
        states_s = states[1:, :-1, :, :H*2].permute(2, 0, 1, 3) # B, T0 T1 H*4
        tmp0, tmp1 = states_s.chunk(2, -1)
        states_s = torch.cat([tmp0.flip(2), tmp1.flip(1)], -1)
        
        return states_s
    
    

        
        
        