
import os, sys
import numpy as np
import six
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def get_optimizer(model, config):
    
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=1e-8, momentum=0.9)
    else:
        raise Exception(f'no such optim: {config.optimizer}')
    return optimizer

def num_of_parameters(m):
    return sum(x.numel() for x in m.parameters())