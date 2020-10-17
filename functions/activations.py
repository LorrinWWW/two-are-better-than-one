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


def squash(s, dim=-1):
    mag_sq = torch.sum(s**2, dim=int(dim), keepdim=True)
    mag = torch.sqrt(mag_sq + 1e-9)
    s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return s

@torch.jit.script
def squash_ts(s, dim=torch.tensor(-1)):
    mag_sq = torch.sum(s**2, dim=int(dim), keepdim=True)
    mag = torch.sqrt(mag_sq + 1e-9)
    s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return s

# def squash(tensor, dim=-1):
#     squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
#     scale = squared_norm / (1 + squared_norm)
#     return scale * tensor / torch.sqrt(squared_norm)