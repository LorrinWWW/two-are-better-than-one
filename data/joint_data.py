

import os, sys
import numpy as np
import torch
import six
import json
import random
import time
import re
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from utils import *
from itertools import combinations, permutations

from .basics import *
from .base import *



### RE

class JointDataLoader(DataLoader):
    
    def __init__(self, json_path, 
                 model=None, num_workers=0, tag_form='iob2', *args, **kargs):
        self.model = model
        self.num_workers = num_workers
        self.dataset = VanillaJsonDataset(json_path)
        self.tag_form = tag_form
        
        super().__init__(dataset=self.dataset, collate_fn=self._collect_fn, num_workers=num_workers, *args, **kargs)
        
        for item in self.dataset.json_list:
            tokens = item['tokens']
            tags = np.zeros(len(tokens), dtype='<U32')
            tags.fill('O')
            for i_begin, i_end, etype in item['entities']:
                tags[i_begin] = f'B-{etype}'
                tags[i_begin+1 : i_end] = f'I-{etype}'
            
            if tag_form == 'iob2':
                item['ner_tags'] = tags
            elif tag_form == 'iobes':
                item['ner_tags'] = BIO2BIOES(tags)
            
            relations = np.zeros([len(tokens), len(tokens)], dtype='<U32')
            relations.fill('O')
            
            for i_begin, i_end, j_begin, j_end, rtype in item['relations']:
            
                relations = self.annotate_relation(relations, i_begin, i_end, j_begin, j_end, f"fw:{rtype}")
                
                # aux annotation
                if relations[j_begin, i_begin] == 'O' or relations[j_begin, i_begin].split(':')[-1] == 'O': 
                    # make sure we dont have conflicts
                    relations = self.annotate_relation(relations, j_begin, j_end, i_begin, i_end, f"bw:{rtype}")
                #else:
                #    print('conflict. ()')
                #    print(relations[i_begin, j_begin], relations[j_begin, i_begin])
                
            item['re_tags'] = relations
        
        if self.num_workers == 0:
            pass # does not need warm indexing
        elif self.model is not None:
            print("warm indexing...")
            tmp = self.num_workers
            self.num_workers = 0
            for batch in self:
                pass
            self.num_workers = tmp
        else:
            print("warn: model is not set, skip warming.")
            print("note that if num_worker>0, vocab will be reset after each batch step,")
            print("thus a warming for indexing is required!")
            
            
    def annotate_relation(self, matrix, i_begin, i_end, j_begin, j_end, rtype):
        matrix[i_begin:i_end, j_begin:j_end] = f"I:{rtype}"
        return matrix
        
        
    def _collect_fn(self, batch):
        tokens, ner_tags, re_tags, relations, entities = [], [], [], [], []
        for item in batch:
            tokens.append(item['tokens'])
            ner_tags.append(item['ner_tags'])
            re_tags.append(item['re_tags'])
            relations.append(item['relations'])
            entities.append(item['entities'])
        
        rets = {
            'tokens': tokens,
            'ner_tags': ner_tags,
            're_tags': re_tags,
            'relations': relations,
            'entities': entities,
        }
        
        if self.model is not None:
            tokens = self.model.token_indexing(tokens)
            ner_tags = self.model.ner_tag_indexing(ner_tags)
            re_tags = self.model.re_tag_indexing(re_tags)
        
            rets['_tokens'] = tokens
            rets['_ner_tags'] = ner_tags
            rets['_re_tags'] = re_tags
        
        return rets
    
    
class JointTrainer(Trainer):
    def __init__(self, train_path, test_path, valid_path,
                 batch_size=128, shuffle=True, model=None, num_workers=0, tag_form='iob2', 
                 *args, **kargs):
        self.batch_size = batch_size
        self.model = model
        self.train = JointDataLoader(train_path, model=model, batch_size=batch_size, 
                                       shuffle=shuffle, num_workers=num_workers, tag_form=tag_form,)
        self.test = JointDataLoader(test_path, model=model, batch_size=8, # small bs for evaluation
                                       num_workers=num_workers, tag_form=tag_form,)
        self.valid = JointDataLoader(valid_path, model=model, batch_size=8, # small bs for evaluation
                                       num_workers=num_workers, tag_form=tag_form,)
    
    # micro f1
    def _get_metrics(self, sent_list, preds_list, labels_list, verbose=0):
        
        n_correct, n_pred, n_label = 0, 0, 0
        i_count = 0
        for sent, preds, labels in zip(sent_list, preds_list, labels_list):
            preds = set(preds)
            labels = {tuple(x) for x in labels}
            
            n_pred += len(preds)
            n_label += len(labels)
            n_correct += len(preds & labels)
                
            i_count += 1
            
        precision = n_correct / (n_pred + 1e-8)
        recall = n_correct / (n_label + 1e-8)
        f1 = 2 / (1/(precision+1e-8) + 1/(recall+1e-8) + 1e-8)

        return precision, recall, f1
    
    def evaluate_model(self, model=None, verbose=0, test_type='valid'):
        
        with torch.no_grad():
            if model is None:
                model = self.model

            if test_type == 'valid':
                g = self.valid
            elif test_type == 'test':
                g = self.test
            else:
                g = []

            sents = []
            pred_entities = []
            pred_relations = []
            pred_relations_wNER = []
            label_entities = []
            label_relations = []
            label_relations_wNER = []
            for i, inputs in enumerate(g):
                inputs = model.predict_step(inputs)
                pred_span_to_etype = [{(ib,ie):etype for ib, ie, etype in x} for x in inputs['entity_preds']]
                label_span_to_etype = [{(ib,ie):etype for ib, ie, etype in x} for x in inputs['entities']]
                pred_entities += inputs['entity_preds']
                label_entities += inputs['entities'] 
                pred_relations += inputs['relation_preds']
                label_relations += inputs['relations']

                pred_relations_wNER += [ 
                    [
                        (ib, ie, m[(ib,ie)], jb, je, m[(jb,je)], rtype) for ib, ie, jb, je, rtype in x
                    ] for x, m in zip(inputs['relation_preds'], pred_span_to_etype)
                ]
                label_relations_wNER += [ 
                    [
                        (ib, ie, m[(ib,ie)], jb, je, m[(jb,je)], rtype) for ib, ie, jb, je, rtype in x 
                    ] for x, m in zip(inputs['relations'], label_span_to_etype)
                ]
                
                sents += inputs['tokens']

            rets = {}
            rets['entity_p'], rets['entity_r'], rets['entity_f1'] = self._get_metrics(
                sents, pred_entities, label_entities, verbose=verbose==1)
            rets['relation_p'], rets['relation_r'], rets['relation_f1'] = self._get_metrics(
                sents, pred_relations, label_relations, verbose=verbose==2)
            rets['relation_p_wNER'], rets['relation_r_wNER'], rets['relation_f1_wNER'] = self._get_metrics(
                sents, pred_relations_wNER, label_relations_wNER, verbose=verbose==3)
        
        return rets
    
    
    def _evaluate_during_train(self, model=None, trainer_target=None, args=None):
        
        if not hasattr(self, 'max_f1'):
            self.max_f1 = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        rets = trainer_target.evaluate_model(model, verbose=0, test_type='test')
        precision, recall, f1 = rets['entity_p'], rets['entity_r'], rets['entity_f1']
        print(f">> test entity prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        precision, recall, f1 = rets['relation_p'], rets['relation_r'], rets['relation_f1']
        print(f">> test relation prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        precision, recall, f1 = rets['relation_p_wNER'], rets['relation_r_wNER'], rets['relation_f1_wNER']
        print(f">> test relation with NER prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        
        rets = trainer_target.evaluate_model(model, verbose=0, test_type='valid')
        precision, recall, f1 = rets['entity_p'], rets['entity_r'], rets['entity_f1']
        e_f1 = f1
        print(f">> valid entity prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        precision, recall, f1 = rets['relation_p'], rets['relation_r'], rets['relation_f1']
        r_f1 = f1
        print(f">> valid relation prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        precision, recall, f1 = rets['relation_p_wNER'], rets['relation_r_wNER'], rets['relation_f1_wNER']
        r_f1_wNER = f1
        print(f">> valid relation with NER prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        
        if e_f1 > self.max_f1[0]:
            self.max_f1[0] = e_f1
            print('new max entity f1 on valid!')
            
        if r_f1 > self.max_f1[1]:
            self.max_f1[1] = r_f1
            print('new max relation f1 on valid!')
            
        if r_f1_wNER > self.max_f1[2]:
            self.max_f1[2] = r_f1_wNER
            print('new max relation f1 with NER on valid!')
            
        if (e_f1 + r_f1) / 2 > self.max_f1[3]:
            self.max_f1[3] = (e_f1 + r_f1) / 2
            print('new max averaged entity f1 and relation f1 on valid!')
            
            if args.model_write_ckpt:
                model.save(args.model_write_ckpt)
                
        if (e_f1 + r_f1_wNER) / 2 > self.max_f1[4]:
            self.max_f1[4] = (e_f1 + r_f1_wNER) / 2
            print('new max averaged entity f1 and relation f1 with NER on valid!')
                
                
class JointTrainerMacroF1(JointTrainer):
    
    # macro f1
    def _get_metrics(self, sent_list, preds_list, labels_list, verbose=0):
        
        label_set = set()
        for labels in labels_list:
            for tmp in labels:
                label_set.add(tmp[-1])
        label_list = sorted(list(label_set))
        
        conf_matrix = np.zeros([len(label_list), 3], dtype=np.float32) # [n_correct, n_label, n_pred]
        for sent, preds, labels in zip(sent_list, preds_list, labels_list):
            preds = set(preds)
            labels = {tuple(x) for x in labels}
            corrects = preds & labels
            
            for tmp in preds:
                if tmp[-1] in label_set:
                    conf_matrix[label_list.index(tmp[-1]), 2] += 1
                else:
                    print('warn: prediction not in label_set, ignore.')
            for tmp in labels:
                conf_matrix[label_list.index(tmp[-1]), 1] += 1
            for tmp in corrects:
                conf_matrix[label_list.index(tmp[-1]), 0] += 1
            
        precision = conf_matrix[:,0] / (conf_matrix[:,2] + 1e-8)
        recall = conf_matrix[:,0] / (conf_matrix[:,1] + 1e-8)
        f1 = 2 / (1/(precision+1e-8) + 1/(recall+1e-8) + 1e-8)

        return precision.mean(0), recall.mean(0), f1.mean(0)
    
    