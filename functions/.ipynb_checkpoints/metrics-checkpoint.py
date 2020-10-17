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


def cosine_similarity(a, b, dim=-1):
    return F.cosine_similarity(a, b, dim=dim)

def cosine_distance(a, b, dim=-1):
    return 1. - cosine_similarity(a, b, dim=dim)

def max_distance(a, b, dim=-1):
    _a, i = a.max(dim=dim)
    _b = b.gather(dim=dim, index=i.unsqueeze(dim)).squeeze(dim)
    return (_a * _b).sqrt() / torch.stack([_a, _b], dim=-1).max(dim=-1)[0]