

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


class GeLU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    
class Mish(nn.Module):
    def forward(self, x):
        return x * (torch.tanh(F.softplus(x))) 
