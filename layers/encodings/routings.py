
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
from layers.encodings import *

        
class SeqDynamicRouting(nn.Module):

    def __init__(self, config, input_dim, C=1, R=1, n_iter=2):
        super().__init__()
        self.config = config
        
        self.C = C # capsules
        self.R = R # route nodes
        self.input_dim = input_dim
        self.output_dim = config.hidden_dim
        self.n_iter = n_iter
        
        self.trans_rnn = LSTMEncoding(self.config, self.input_dim)
        
#         self.W = nn.Parameter(torch.randn(
#             self.C, 1, self.R, self.output_dim, self.output_dim
#         ))
#         bias = np.sqrt(6.0 / (self.output_dim + self.output_dim))
#         nn.init.uniform_(self.W.data, -bias, bias)

        self.dense = nn.Linear(self.output_dim, self.C * self.output_dim, bias=False)
        bias = np.sqrt(6.0 / (self.output_dim + self.output_dim))
        nn.init.uniform_(self.dense.weight.data, -bias, bias)
        
#         self.b = nn.Parameter(torch.randn(
#             self.C, 1, 1, self.R, self.input_dim
#         ))
#         bias = np.sqrt(6.0 / (1 + self.input_dim))
#         nn.init.uniform_(self.b.data, -bias, bias)
        
    def forward(self, inputs, mask=None):
        B, T, R, input_dim = inputs.shape # (B, T, R, in)
        C, output_dim = self.C, self.output_dim
        
        u = inputs.permute(0, 2, 1, 3).reshape(B*R, T, input_dim) # (BR, T, in)
        u = self.trans_rnn(u, False) # (BR, T, out)
#         u = u.view(B, R, T, output_dim).permute(0, 2, 1, 3).reshape(1, B*T, R, 1, output_dim) # (1, BT, R, 1, out)
#         u = u @ self.W # (C, B, R, 1, out)
        u = self.dense(u) # (BR, T, C*out)
        u = u.view(B, R, T, C, 1, output_dim).permute(3, 0, 2, 1, 4, 5) # (C, B, T, R, 1, out)
        u = u.reshape(C, B*T, R, 1, output_dim) # (C, BT, R, 1, out)

#         u = inputs[None, :, :, :, :] + self.b # (C, B, T, R, H)
#         u = u.permute(0, 1, 3, 2, 4).reshape(C*B*R, T, input_dim) # (CBR, T, in)
#         u = self.trans_rnn(u, False) # (CBR, T, out)
#         u = u.view(C, B, R, T, output_dim).permute(0, 1, 3, 2, 4).reshape(C, B*T, R, 1, output_dim) # (1, BT, R, 1, out)
            
        if mask is not None:
            float_mask = mask.float()[None, :, :, None, None].repeat(1, T, 1, 1, 1) # (B, R) => (1, B, R, 1, 1)
        else:
            float_mask = 1.
        
        b = torch.zeros(C, B*T, R, 1, 1).to(inputs.device) # (C, B, R, 1, 1)
        for i in range(self.n_iter):
            v = b.softmax(dim=0) * float_mask # (C, B, R, 1, 1)
            c_hat = (u * v).sum(2, keepdims=True) # (C, B, 1, 1, out)
            c = squash(c_hat, dim=-1) # (C, B, 1, 1, out)
            b = b + (c * u).sum(-1, keepdims=True) # (C, B, R, 1, 1)
        
#         out_c_hat = c_hat[:, :, 0, 0, :].permute(1, 0, 2) # (B, C, out)
        out_c = c[:, :, 0, 0, :].permute(1, 0, 2) # (B, C, out)
        out_v = v[:, :, :, 0, 0].permute(1, 2, 0) # (B, R, C)
        out_b = b[:, :, :, 0, 0].permute(1, 2, 0) # (B, R, C)
        
        return out_c, out_v, out_b

    
class DynamicRouting(nn.Module):

    def __init__(self, config, input_dim, C=1, R=1, n_iter=2):
        super().__init__()
        self.config = config
        
        self.C = C # capsules
        self.R = R # route nodes
        self.input_dim = input_dim
        self.output_dim = config.hidden_dim
        self.n_iter = n_iter
        
        self.dense = nn.Linear(self.input_dim, self.C * self.output_dim, bias=False)
        bias = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        nn.init.uniform_(self.dense.weight.data, -bias, bias)
        
        
    def forward(self, inputs, mask=None):
        B, R, input_dim = inputs.shape # (B, R, in)
        C, output_dim = self.C, self.output_dim
        
        u = self.dense(inputs) # (B, R, C*out)
        u = u.view(B, R, C, 1, output_dim).permute(2, 0, 1, 3, 4) # (C, B, T, R, 1, out)
            
        if mask is not None:
            float_mask = mask.float()[None, :, :, None, None] # (B, R) => (1, B, R, 1, 1)
        else:
            float_mask = 1.
        
        b = torch.zeros(C, B, R, 1, 1).to(inputs.device) # (C, B, R, 1, 1)
        for i in range(self.n_iter):
            v = b.softmax(dim=0) * float_mask # (C, B, R, 1, 1)
            c_hat = (u * v).sum(2, keepdims=True) # (C, B, 1, 1, out)
            c = squash(c_hat, dim=-1) # (C, B, 1, 1, out)
            b = b + (c * u).sum(-1, keepdims=True) # (C, B, R, 1, 1)
        
#         out_c_hat = c_hat[:, :, 0, 0, :].permute(1, 0, 2) # (B, C, out)
        out_c = c[:, :, 0, 0, :].permute(1, 0, 2) # (B, C, out)
        out_v = v[:, :, :, 0, 0].permute(1, 2, 0) # (B, R, C)
        out_b = b[:, :, :, 0, 0].permute(1, 2, 0) # (B, R, C)
        
        return out_c, out_v, out_b
    
    
class SupportRouting(nn.Module):
 
    def __init__(self, config, input_dim, n_iter=2):
        super().__init__()
        
        self.config = config
        
        self.input_dim = input_dim
        self.output_dim = config.hidden_dim
        self.n_iter = n_iter

    
        self.dense_out = nn.Sequential(
            nn.Linear(2*input_dim, self.output_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim, bias=False),
#             nn.ReLU(),
#             nn.Linear(self.output_dim, self.output_dim, bias=False),
        )
        init_linear(self.dense_out[0])
        init_linear(self.dense_out[2])
#         init_linear(self.dense_out[4])
        
    def forward(self, inputs, caps, mask=None):
        B, R, input_dim = inputs.shape # (B, R, in)
        C, output_dim = caps.shape[1], self.output_dim #

        inputs = inputs[None, :, :, None, :] # (1, B, R, 1, in)
        caps = caps[None, :, :, None, :].transpose(0, 2) # (C, B, 1, 1, in)
        inputs, caps = torch.broadcast_tensors(inputs, caps) # (C, B, R, 1, in)
        u = torch.cat([inputs, caps], axis=-1) # (C, B, R, 1, 2*in)
        u = self.dense_out(u) # (C, B, R, 1, out)
        
        if mask is not None:
            float_mask = mask.float()[None, :, :, None, None] # (B, R) => (1, B, R, 1, 1)
        else:
            float_mask = 1.
        
        b = torch.zeros(C, B, R, 1, 1).to(inputs.device) # (C, B, R, 1, 1)
        for i in range(self.n_iter):
            v = b.softmax(dim=0) * float_mask # (C, B, R, 1, 1)
            c_hat = (u * v).sum(2, keepdims=True) # (C, B, 1, 1, out)
            c = squash(c_hat, dim=-1) # (C, B, 1, 1, out)
            b = b + (c * u).sum(-1, keepdims=True) # (C, B, R, 1, 1)
        
        out_c = c[:, :, 0, 0, :].permute(1, 0, 2) # (B, C, out)
        out_v = v[:, :, :, 0, 0].permute(1, 2, 0) # (B, R, C)
        out_b = b[:, :, :, 0, 0].permute(1, 2, 0) # (B, R, C)
        
#         print(out_v[0, 0])
        
        return out_c, out_v, out_b
    
class SelfRouting(SupportRouting):
        
    def forward(self, inputs, mask=None):
        return super().forward(inputs, inputs, mask)