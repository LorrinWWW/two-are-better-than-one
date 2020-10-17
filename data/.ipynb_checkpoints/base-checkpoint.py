
import os, sys
import numpy as np
import torch
import six
import json
import random
import time
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from utils import *
from itertools import combinations 


class Trainer:
    
    def __iter__(self, *args, **kargs):
        return self.train.__iter__(*args, **kargs)
    
    @must_override
    def evaluate_model(self):
        pass
        
        
    @warn_not_override
    def _evaluate_during_train(self, model=None, trainer_target=None, args=None):
        pass
        
    def train_model(self, model=None, trainer_target=None, args=None):
        
        if model is None:
            model = self.model
        
        if args is None:
            raise Exception('require args')
            
        trainer_source = self
        if trainer_target is None:
            trainer_target = self
        
        losses = []
        times = []
        decay_rate = args.decay_rate
        learning_rate = args.lr
        for i_epoch in range(args.max_epoches):

            global_steps = int(model.global_steps.data)

            if global_steps > args.max_steps:
                print(f"reach max_steps, stop training")
                break

            tic = time.time()
            for i, batch in enumerate(trainer_source):
                loss = model.train_step(batch)['loss'].detach().cpu().numpy()
                losses.append(loss)
                toc = time.time()
                times.append(toc - tic)

                global_steps = int(model.global_steps.data)
                if global_steps % 100 == 0:
                    print(f"g_step {global_steps}, step {i+1}, "
                          f"avg_time {sum(times)/len(times):.3f}, "
                          f"loss:{sum(losses)/len(losses):.4f}")
                    losses = []
                    times = []

                tic = time.time()

                if global_steps % 1000 == 0:
                    _lr = learning_rate/(1+decay_rate*global_steps/1000)
                    print(f"learning rate was adjusted to {_lr}")
                    adjust_learning_rate(model.optimizer, lr=_lr)

                if global_steps % args.evaluate_interval == 0:
                    self._evaluate_during_train(model=model, trainer_target=trainer_target, args=args)

                if global_steps == args.max_steps:
                    break