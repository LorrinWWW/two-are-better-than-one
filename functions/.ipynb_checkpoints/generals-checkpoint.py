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
