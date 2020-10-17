
import os, sys
import numpy as np
import torch
import six
import json
import csv
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset
from utils import *
from itertools import combinations, cycle


class RowIterDataset(IterableDataset):
    
    def __init__(self, path):
        self.path = path
        
    def _parse_file(self, path):
        with open(path) as f:
            for row in f:
                yield row.strip()

    def _get_stream(self, path):
        return cycle(self._parse_file(path))
   
    def __iter__(self):
        return self._get_stream(self.path)
        

class RowDataset(Dataset):
    '''read file by lines and shuffle it, that's all.'''
    def __init__(self, path):
        self.path = path
        with open(path, 'r') as f:
            self.rows = [row.strip() for row in f]
            
        random.shuffle(self.rows)
        
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        return self.rows[idx]


class VanillaJsonDataset(Dataset):
    '''read json and shuffle it, that's all.'''
    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.json_list = json.load(f)
        
        random.shuffle(self.json_list)
        
    def __len__(self):
        return len(self.json_list)
    
    def __getitem__(self, idx):
        return self.json_list[idx]


class SimpleJsonDataset(Dataset):
    
    def __init__(self, json_path, tokenizer, tag_form='iob2', skip_empty=False):
        self.json_path = json_path
        self.tag_form = tag_form.lower()
        self.tokenizer = tokenizer
        with open(json_path, 'r') as f:
            self.json_list = [item for item in json.load(f) if len(item['tokens'])>0]
            
        for item in self.json_list:
            tokens, _, spans = convert_to_new_length(item['tokens'], ['O']*len(item['tokens']), self.tokenizer)
            item['original_tokens'] = item['tokens']
            item['original_entities'] = item['entities']
            item['tokens'] = tokens
            item['entities'] = [
                {
                    'entity_type': e['entity_type'], 
                    'span': (spans[e['span'][0]][0], spans[e['span'][1]][1])
                } for e in item['original_entities']
            ]
            
        print(f"load from {json_path}")
        print(f"{len(self.json_list)} valid sentences.")
        
        random.shuffle(self.json_list)
        
    def __len__(self):
        return len(self.json_list)
    
    def __getitem__(self, idx):
        return self.json_list[idx]


class SimpleJsonDataset(Dataset):
    
    def __init__(self, json_path, tag_form='iob2', skip_empty=False):
        self.json_path = json_path
        self.tag_form = tag_form.lower()
        with open(json_path, 'r') as f:
            self.json_list = [item for item in json.load(f) if len(item['tokens'])>0]
            
        print(f"load from {json_path}")
        print(f"{len(self.json_list)} valid sentences.")
        
        random.shuffle(self.json_list)
        
    def __len__(self):
        return len(self.json_list)
    
    def __getitem__(self, idx):
        return self.json_list[idx]

class JsonDataset(Dataset):
    
    def __init__(self, json_path, tag_form='iob2', skip_empty=False):
        self.json_path = json_path
        self.tag_form = tag_form.lower()
        with open(json_path, 'r') as f:
            self.json_list = [item for item in json.load(f) if len(item['tokens'])>0]
        if skip_empty:
            self.json_list = [item for item in self.json_list if not all(t=='O' for t in item['slot_tags'])]
            
        for item in self.json_list:
            tags = item['slot_tags']
            tags = ALL2BIO(tags)
            if self.tag_form == 'iobes':
                tags = BIO2BIOES(tags)
            elif self.tag_form == 'iob2':
                pass
            else:
                raise Exception(f"no such tag form: {self.tag_form}.")
            
        print(f"load from {json_path}")
        print(f"{len(self.json_list)} valid sentences.")
        
        random.shuffle(self.json_list)
        
    def __len__(self):
        return len(self.json_list)
    
    def __getitem__(self, idx):
        return self.json_list[idx]
    
    
class SubsetClassesJsonDataset(Dataset):
    
    def __init__(self, json_path, tag_form='iob2', skip_empty=False, num_sampled_classes=5, sampling=lambda x: x):
        self.json_path = json_path
        self.tag_form = tag_form.lower()
        with open(json_path, 'r') as f:
            self.json_list = [item for item in json.load(f) if len(item['tokens'])>0]
        if skip_empty:
            self.json_list = [item for item in self.json_list if not all(t=='O' for t in item['slot_tags'])]
        self.json_list = sampling(self.json_list)
            
        for item in self.json_list:
            tags = item['slot_tags']
            tags = ALL2BIO(tags)
            if self.tag_form == 'iobes':
                tags = BIO2BIOES(tags)
            elif self.tag_form == 'iob2':
                pass
            else:
                raise Exception(f"no such tag form: {self.tag_form}.")
            
        print(f"load from {json_path}")
        print(f"{len(self.json_list)} valid sentences.")
        
        # for sampling
        self.num_sampled_classes = num_sampled_classes
        self.cls2idxs = defaultdict(list)
        for i, item in enumerate(self.json_list):
            self.cls2idxs[item['category']].append(i)
        self.update_sampled_classes()
        
        random.shuffle(self.json_list)
        
    def update_sampled_classes(self):
        self.sampled_classes = random.sample(self.cls2idxs.keys(), self.num_sampled_classes)
        self.idxs = sum([self.cls2idxs[c] for c in self.sampled_classes], [])
        
    def __len__(self):
        return len(self.json_list)
    
    def __getitem__(self, idx):
        idx = random.choice(self.idxs)
        return self.json_list[idx]