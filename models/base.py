import os, sys, pickle
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
from layers import *
from functions import *
from . import Config

import copy

class Tagger(nn.Module):
    
    def __init__(self, config,):
        super().__init__()
        
        self.config = config
        
        self.before_init()
        
        self.set_embedding_layer()
        self.set_encoding_layer()
        self.set_logits_layer()
        self.set_loss_layer()
        self.set_training_related()
        
        # hook
        self._train_step = self.train_step
        self.train_step = self.hooked_train_step
        
        self._predict_step = self.predict_step
        self.predict_step = self.hooked_predict_step
        
        self.check_attrs()
        
        # put to device
        self.device = self.config.device
        self.to(self.device)
        
        self.after_init()
    
    def before_init(self):
        pass
    
    def after_init(self):
        pass
        
    def check_attrs(self):
        # indexing
        assert hasattr(self, 'tag_indexing')
        assert hasattr(self, 'token_indexing')
        
    ### STRUCTURE
    @must_override
    def set_embedding_layer(self):
        pass
        
    @must_override
    def set_encoding_layer(self):
        pass
    
    @must_override
    def set_logits_layer(self):
        pass
    
    @must_override
    def set_loss_layer(self):
        pass
    
    def set_training_related(self):
        self.global_steps = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.optimizer = get_optimizer(self, self.config)
        
    @warn_not_override
    def get_default_trainer_class(self):
        raise Exception('undefined.')
    
    ### BEHAVIOUR
    @must_override
    def forward(self, inputs):
        pass
    
    @must_override
    def forward_step(self, inputs):
        pass
    
    @must_override
    def predict_step(self, inputs):
        pass
    
    def hooked_predict_step(self, inputs):
        self.eval()
        rets = self._predict_step(inputs)
        return rets
    
    @must_override
    def train_step(self, inputs):
        pass
    
    def hooked_train_step(self, inputs):
        grad_period = self.config.grad_period if hasattr(self.config, 'grad_period') else 1
        self.train()
        if grad_period == 1 or self.global_steps.data % grad_period == 0:
            self.zero_grad()
        rets = self._train_step(inputs)
        if grad_period == 1 or self.global_steps.data % grad_period == grad_period-1:
            self.optimizer.step()
        self.global_steps.data += 1
        return rets
    
#     def hooked_train_step(self, inputs):
#         self.train()
#         self.zero_grad()
#         rets = self._train_step(inputs)
#         self.optimizer.step()
#         self.global_steps.data += 1
#         return rets

    ### SAVE
    @warn_not_override
    def save_ckpt(self, path):
        torch.save(self.state_dict(), path+'.pt')

    def save(self, path):
        self.save_ckpt(path)
        with open(path+'.json', 'w') as f:
            json.dump(self.config.__dict__, f)
        
    @warn_not_override       
    def load_ckpt(self, path):
        self.load_state_dict(torch.load(path+'.pt'))
        
    @classmethod
    def load(cls, path):
        with open(path+'.json', 'r') as f:
            config = Config(**json.load(f))
        obj = cls(config)
        obj.load_ckpt(path)
        return obj
        
    
    
class Joint(Tagger):
    
    def check_attrs(self):
        # indexing
        assert hasattr(self, 'tag_indexing')
        assert hasattr(self, 'token_indexing')
        assert hasattr(self, 'cls_indexing')