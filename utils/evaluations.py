
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

def is_overlapping(span_a, span_b):
    if span_a[0] <= span_b[0] < span_a[1] or span_a[0] < span_b[1] <= span_a[1]:
        return True
    span_a, span_b = span_b, span_a
    if span_a[0] <= span_b[0] < span_a[1] or span_a[0] < span_b[1] <= span_a[1]:
        return True
    return False

def is_nesting(span_a, span_b):
    if span_a[0] <= span_b[0] < span_a[1] and span_a[0] < span_b[1] <= span_a[1]:
        return True
    span_a, span_b = span_b, span_a
    if span_a[0] <= span_b[0] < span_a[1] and span_a[0] < span_b[1] <= span_a[1]:
        return True
    return False

def is_overlapping_list(span_a, span_list):
    for span_b in span_list:
        if is_overlapping(span_a, span_b):
            return True
    return False
        
def has_overlapping_but_not_nested(l):
    for last, curr in combinations(l, 2):
        if (curr[2][0] < last[2][0] < curr[2][1] and last[2][1] > curr[2][1]) or \
            (last[2][0] < curr[2][0] < last[2][1] and curr[2][1] > last[2][1]):
            return True
    return False

def has_overlapping(l):
    for last, curr in combinations(l, 2):
        if is_overlapping(last[2], curr[2]):
            return True
    return False