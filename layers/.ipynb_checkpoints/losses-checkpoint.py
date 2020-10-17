
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
    
    
class TwoWayLoss(nn.Module):
    
    def __init__(self, reduction='sum', lambd_l2p=0.8, lambd_p2l=0.2, use_softmin=False):
        super().__init__()
        
        self.use_softmin = use_softmin
        self.reduction = reduction
        self.lambd_l2p = lambd_l2p
        self.lambd_p2l = lambd_p2l
        
        self.ce_layer = nn.CrossEntropyLoss(reduction='none')
        
    def min(self, t, dim):
        
        if self.use_softmin:
            
            return (t * F.softmax(-t, dim=dim)).sum(dim=dim)
        else:
            return t.min(dim=dim).values
    
    def forward(self, logits, labels, label_masks):
        '''
        logits (B, T, P, C, M)
        labels (B, T, L, A, C)
        label_masks (B, L, A, C)
        
        两个min是核心思想
        '''
        B, T, P, C, M = logits.shape
        B, T, L, A, C = labels.shape
        
        logits = logits.view(B, T, 1, 1, P, C, M).repeat(1, 1, L, A, 1, 1, 1)
        labels = labels.view(B, T, L, A, 1, C).repeat(   1, 1, 1, 1, P, 1,)
        
        loss_tensor = self.ce_layer(logits.permute(0,-1,1,2,3,4,5), labels).sum(1) # (B, L, A, P, C)
        loss_masks = label_masks.view(B, L, A, 1, C).float() # 1 for valid, 0 for pads 
        masked_loss_tensor = loss_tensor + 1e6*(1.-loss_masks)
        
        #l2p_loss = self.get_l2p_loss(loss_dict, logits, labels)
        all_label_masks = label_masks.any(2).view(B, L, C).float() # (B, L, C)
        l2p_loss_tensor = self.min(masked_loss_tensor.view(B, L, A*P, C), dim=2) # (B, L, C)
        
        l2p_loss = (l2p_loss_tensor * all_label_masks).sum() # 对于一个entity，所有的排列组合都被mask了，说明它是pads
        
#         print(l2p_loss)
        
        #p2l_loss = self.get_p2l_loss(loss_dict, logits, labels)
        empty_labels = torch.zeros([B, T, 1, P, C], dtype=torch.long).to(logits.device)
        empty_loss_tensor = self.ce_layer(logits[:, :, :1, 0, :, :, :].permute(0,-1,1,2,3,4), empty_labels).sum(1) # (B, 1, P, C)
        p2l_masked_loss_tensor = masked_loss_tensor.view(B, L*A, P, C)
        p2l_masked_loss_tensor = torch.cat([p2l_masked_loss_tensor, empty_loss_tensor], dim=1) # (B, L*A+1, P, C)
        l2p_loss_tensor = self.min(p2l_masked_loss_tensor, dim=1) # (B, P, C)
        p2l_loss = l2p_loss_tensor.sum()
        
#         print(p2l_loss)
        
        loss = self.lambd_l2p * l2p_loss + self.lambd_p2l * p2l_loss
        
        return loss