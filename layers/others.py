
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
    
    
class Rehapse(nn.Module):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape
        
    def forward(self, x, shape=None):
        if shape is None:
            shape = self.shape
        return x.reshape(shape)
    

class View(nn.Module):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape
        
    def forward(self, x, shape=None):
        if shape is None:
            shape = self.shape
        return x.view(shape)
    
class Permute(nn.Module):
    def __init__(self, order=None):
        super().__init__()
        self.order = order
        
    def forward(self, x, order=None):
        if order is None:
            order = self.order
        return x.permute(order)