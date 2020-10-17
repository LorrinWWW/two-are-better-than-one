
import os, sys
import numpy as np
import six
import json
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gpustat

def get_available_gpu_memory_list():
    ret = gpustat.new_query()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible_devices = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        return [gpu.memory_available for i, gpu in enumerate(ret.gpus) if i in visible_devices]
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

def wait_util_enough_mem(min_mem, sleep_time=5, max_n_try=None):
    n_try = 0
    while True:
        n_try += 1
        gpu_idx, gpu_avail_mem = get_max_available_gpu()
        if gpu_avail_mem >= min_mem:
            return gpu_idx
        if max_n_try is not None and n_try >= max_n_try:
            return None
        time.sleep(sleep_time)