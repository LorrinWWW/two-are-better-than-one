import os, sys
import six
import random
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

@torch.jit.script
def to_one_hot(y, N):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y#.data if isinstance(y, torch.autograd.Variable) else y
    y_tensor = y_tensor.long().view(-1, 1)
    if int(N) <= 0:
        N = torch.max(y_tensor).long() + 1
    y_one_hot = torch.zeros(y_tensor.shape[0], N).to(y_tensor.device).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(y.shape + (-1,))
    return y_one_hot

@torch.jit.script
def gather_by_tags(x, tags):
    tags = to_one_hot(tags, N=torch.tensor(0))
    centers = tags.transpose(0, 1) @ x
    centers = centers / (tags.sum(0)[:, None] + 1e-8)
    return centers

def max_tensors(tensors):
    tensors = torch.broadcast_tensors(*tensors)
    out, _ = torch.stack(tensors, -1).max(-1)
    return out


def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

def pad_arbitrary(sequences, padding_value=0., padding_dim=0, length_dim=1):
    max_size = list(sequences[0].size())
    max_len = max_len = max([s.size(length_dim) for s in sequences])
    max_size[length_dim] = max_len
    out_dims = max_size.copy()
    out_dims.insert(padding_dim, len(sequences))
    
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    
    for i, tensor in enumerate(sequences):
        length = tensor.size(length_dim)
        out_tensor.select(padding_dim, i).narrow(length_dim, 0, length)[:] = tensor
    
    return out_tensor
