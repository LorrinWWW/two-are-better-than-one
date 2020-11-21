#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from utils import *
from data import *
from models import *

# torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)

# In[3]:


import argparse

def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

parser = argparse.ArgumentParser(description='Arguments for training.')

#### Train
parser.add_argument('--model_class',
                    default='None',
                    action='store',)

parser.add_argument('--model_read_ckpt',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--model_write_ckpt',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--pretrained_wv',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--dataset',
                    default='ACE05',
                    action='store',)

parser.add_argument('--label_config',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--batch_size',
                    default=32, type=int,
                    action='store',)

parser.add_argument('--evaluate_interval',
                    default=1000, type=int,
                    action='store',)

parser.add_argument('--max_steps',
                    default=int(1e9), type=int,
                    action='store')

parser.add_argument('--max_epoches',
                    default=100, type=int,
                    action='store')

parser.add_argument('--decay_rate',
                    default=0.05, type=float,
                    action='store')


#### Model Config
parser.add_argument('--token_emb_dim',
                    default=100, type=int,
                    action='store',)

parser.add_argument('--char_encoder',
                    default='lstm',
                    action='store',)

parser.add_argument('--char_emb_dim',
                    default=0, type=int,
                    action='store',)

parser.add_argument('--cased',
                    default=False, type=int,
                    action='store',)

parser.add_argument('--hidden_dim',
                    default=200, type=int,
                    action='store',)

parser.add_argument('--num_layers',
                    default=3, type=int,
                    action='store',)

parser.add_argument('--crf',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--loss_reduction',
                    default='sum',
                    action='store',)

parser.add_argument('--maxlen',
                    default=None, type=int,
                    action='store',)

parser.add_argument('--dropout',
                    default=0.5, type=float,
                    action='store',)

parser.add_argument('--optimizer',
                    default='sgd',
                    action='store',)

parser.add_argument('--lr',
                    default=0.02, type=float,
                    action='store',)

parser.add_argument('--vocab_size',
                    default=500000, type=int,
                    action='store',)

parser.add_argument('--vocab_file',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--ner_tag_vocab_size',
                    default=64, type=int,
                    action='store',)

parser.add_argument('--re_tag_vocab_size',
                    default=128, type=int,
                    action='store',)

parser.add_argument('--lm_emb_dim',
                    default=0, type=int,
                    action='store',)

parser.add_argument('--lm_emb_path',
                    default='', type=str,
                    action='store',)

parser.add_argument('--head_emb_dim',
                    default=0, type=int,
                    action='store',)

parser.add_argument('--tag_form',
                    default='iob2',
                    action='store',)

parser.add_argument('--warm_steps',
                    default=1000, type=int,
                    action='store',)

parser.add_argument('--grad_period',
                    default=1, type=int,
                    action='store',)

parser.add_argument('--device',
                    default=None, type=none_or_str,
                    action='store',)


# In[4]:

args = parser.parse_args()


# In[5]:


if args.device is not None:
    torch.cuda.set_device(args.device)
else:
    gpu_idx, gpu_mem = set_max_available_gpu()
    args.device = f"cuda:{gpu_idx}"


# In[6]:


config = Config(**args.__dict__)
ModelClass = eval(args.model_class)
model = ModelClass(config)


# In[7]:


if args.model_read_ckpt:
    print(f"reading params from {args.model_read_ckpt}")
    model = model.load(args.model_read_ckpt)
    model.token_embedding.token_indexing.update_vocab = False
elif args.token_emb_dim > 0 and args.pretrained_wv:
    print(f"reading pretrained wv from {args.pretrained_wv}")
    model.token_embedding.load_pretrained(args.pretrained_wv, freeze=True)
    model.token_embedding.token_indexing.update_vocab = False


# In[8]:


print("reading data..")
Trainer = model.get_default_trainer_class()
flag = args.dataset
trainer = Trainer(
    model=model,
    train_path=f'./datasets/unified/train.{flag}.json',
    test_path=f'./datasets/unified/test.{flag}.json',
    valid_path=f'./datasets/unified/valid.{flag}.json',
    label_config=args.label_config,
    batch_size=int(args.batch_size),
    tag_form=args.tag_form, num_workers=0,
)


# In[ ]:

# trainer.evaluate_model()


# %%capture cap
print("=== start training ===")
trainer.train_model(args=args)




