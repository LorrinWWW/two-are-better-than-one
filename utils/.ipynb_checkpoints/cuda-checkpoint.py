
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

import gpustat

def get_available_gpu_memory_list():
    ret = gpustat.new_query()
    return [gpu.memory_available for gpu in ret.gpus]

def get_max_available_gpu():
    gpu_available_memory_list = get_available_gpu_memory_list()
    gpu_idx = int(np.argmax(gpu_available_memory_list))
    gpu_mem = gpu_available_memory_list[gpu_idx]
    return gpu_idx, gpu_mem

def set_max_available_gpu():
    gpu_idx, gpu_mem = get_max_available_gpu()
    torch.cuda.set_device(gpu_idx)
    return gpu_idx, gpu_mem