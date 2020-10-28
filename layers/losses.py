
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


class SparseLoss(nn.Module):
    
    def __init__(self, reduction='sum', norm='softmax', *args, **kargs):
        super().__init__()
        self.reduction = reduction
        self.norm = norm
        self.loss_layer = self.get_loss_class()(reduction=reduction, *args, **kargs)
    
    def get_loss_class(self):
        raise Exception('not implemented')
        
    def forward(self, logits, labels):
        if self.norm == 'softmax':
            logits = logits.softmax(-1)
        depth = torch.tensor(logits.shape[-1])
        labels = to_one_hot(labels, depth)
        return self.loss_layer(logits, labels)
    
class SparseCE(nn.Module):
    def __init__(self, reduction='sum', *args, **kargs):
        super().__init__()
        self.ce_layer = nn.CrossEntropyLoss(reduction=reduction, *args, **kargs)
    
    def forward(self, logits, labels):
        if len(logits.shape) == 3:
            logits = logits.transpose(1, 2)
        elif len(logits.shape) > 3:
            raise Exception('not defined')
        return self.ce_layer(logits, labels)
    
class SparseMSE(SparseLoss):
    def get_loss_class(self):
        return nn.MSELoss
    
class SparseMME(SparseLoss):
    def get_loss_class(self):
        self.norm = ''
        return MMELoss
    
class MMELoss(nn.Module):
    '''Max-margin Loss'''
    def __init__(self, reduction='sum', m_plus=0.9, m_minus=0.1, lambd=0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambd = lambd
        self.reduction = reduction
        
    def forward(self, logits, labels):
        loss = labels * F.relu(self.m_plus - logits).pow(2) + self.lambd * (1. - labels) * F.relu(logits - self.m_minus).pow(2)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        else:
            pass
        return loss
    
class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=None, reduction='sum'):
        '''
        0 <= gamma <= 5
        '''
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        shape_ckpt = target.shape
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.reduction == 'mean': 
            return loss.mean()
        elif self.reduction == 'sum': 
            return loss.sum()
        elif self.reduction == 'none':
            return loss.view(shape_ckpt)
        else:
            raise Exception(f'{self.reduction} ??')
            
class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.0, reduction='sum'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, input, target):
        
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        shape_ckpt = target.shape
        target = target.view(-1, 1)
        
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target, (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1)
        
        if self.reduction == 'mean': 
            return loss.mean()
        elif self.reduction == 'sum': 
            return loss.sum()
        elif self.reduction == 'none':
            return loss.view(shape_ckpt)
        else:
            raise Exception(f'{self.reduction} ??')