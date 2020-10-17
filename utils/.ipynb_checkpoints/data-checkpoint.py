
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

def count_tag_nums(json_list, count_tags={'B', 'I'}):
    count = defaultdict(int)
    for item in json_list:
        for tag in item['slot_tags']:
            if tag[0] in count_tags:
                count[tag[:]] += 1
    return count

def sample_k_shot_slot_filling(k, json_list, count_tags={'B', 'I'}, seed=None):
    '''
    k-shot
    domain D
    label set LD
    '''
    if seed:
        random.seed(seed)

    D = {tuple(item['tokens']):item for item in json_list}
    D_keys = set(D)
    D_label_keys = defaultdict(set)
    key_labels = {}
    LD = set()
    for item in json_list:
        key = tuple(item['tokens'])
        labels = [tag[:] for tag in item['slot_tags'] if tag[0] in count_tags]
        key_labels[key] = labels
        LD.update(labels)
        for label in set(labels):
            D_label_keys[label].add(key)
        
    S_keys = set()
    count = {l: 0 for l in LD}
    
    all_count = count_tag_nums(json_list, count_tags=count_tags)
    for l in count:
        n = all_count.get(l, 0)
        if n < k:
            count[l] += k-n
    
    # sample
    for l in sorted(list(LD)):
        while count[l] < k:
            tmp = sorted(list(D_label_keys[l] - S_keys))
            if len(tmp) == 0:
                break
            key = random.choice(tmp)
            S_keys.add(key)
            for lj in key_labels[key]:
                count[lj] += 1
                    
    # remove
    for key in sorted(list(S_keys)):
        S_keys.remove(key)
        for lj in key_labels[key]:
            count[lj] -= 1
        if any(v<k for v in count.values()):
            S_keys.add(key)
            for lj in key_labels[key]:
                count[lj] -= 1
                
    S = [D[k] for k in S_keys]
    return S

def get_seq_metrics(sents, labels, preds, verbose=0):
    n_correct = n_recall = n_precision = 0
    confusion_dict = defaultdict(lambda: [0, 0, 0]) # n_correct, n_preds, n_labels
    for i in range((len(sents))):
        i_label = labels[i]
        i_pred = preds[i][:len(i_label)]
        i_sent = sents[i]

        spans, types = tag2span(i_pred, True, True)
        pred_set = {(_type, _span[0], _span[1]) for _span, _type in zip(spans, types)}

        spans, types = tag2span(i_label, True, True)
        label_set = {(_type, _span[0], _span[1]) for _span, _type in zip(spans, types)}

        correct_set = pred_set & label_set
        
        for _type, _, _ in correct_set:
            confusion_dict[_type][0] += 1
        for _type, _, _ in pred_set:
            confusion_dict[_type][1] += 1
        for _type, _, _ in label_set:
            confusion_dict[_type][2] += 1

        n_correct += len(correct_set)
        n_recall += len(label_set)
        n_precision += len(pred_set)

        if verbose > 0:
            not_recall = label_set - correct_set
            not_precise = pred_set - correct_set
            if not_recall or not_precise:
                print('===')
                for category, span_i, span_j in not_recall:
                    print(' '.join(i_sent[span_i:span_j]), category)
                print('--')
                for category, span_i, span_j in not_precise:
                    print(' '.join(i_sent[span_i:span_j]), category)
                print('===')
                
    if verbose > 0:
        print(n_correct, n_precision, n_recall)
    
    
    try:
        recall = n_correct / n_recall
        precision = n_correct / n_precision
        f1 = 2 / (1/recall + 1/precision)
    except:
        recall = precision = f1 = 0
        
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_dict': confusion_dict,
    }

def get_cls_metrics(sents, labels, preds, verbose=0):
    n_correct = n_total = 0
    confusion_dict = defaultdict(lambda: [0, 0, 0]) # n_correct, n_preds, n_labels
    for i in range((len(sents))):
        i_label = labels[i]
        i_pred = preds[i]
        i_sent = sents[i]
        
        n_total += 1
        confusion_dict[i_label][2] += 1
        confusion_dict[i_pred][1] += 1
        
        if i_label == i_pred:
            n_correct += 1
            confusion_dict[i_pred][0] += 1
    
    try:
        acc = n_correct / n_total
    except:
        acc = 0
        
    return {
        'acc': acc,
        'confusion_dict': confusion_dict,
    }
    
    
def ALL2BIO(tags):
    ret = []
    for tag in tags:
        if tag[0] == 'S':
            ret.append('B'+tag[1:])
        elif tag[0] == 'E':
            ret.append('I'+tag[1:])
        else:
            ret.append(tag)
    return ret

def BIO2BIOE(tags):
    ret = []
    for i in range(len(tags)):
        if tags[i][0] == 'I' and (i+1==len(tags) or tags[i+1]!=tags[i]):
            ret.append('E'+tags[i][1:])
        else:
            ret.append(tags[i])
    return ret

def BIO2BIOES(tags):
    ret = []
    for i in range(len(tags)):
        if tags[i][0] == 'I' and (i+1==len(tags) or tags[i+1]!=tags[i]):
            ret.append('E'+tags[i][1:])
        elif tags[i][0] == 'B' and (i+1==len(tags) or tags[i+1][0]!='I'):
            ret.append('S'+tags[i][1:])
        else:
            ret.append(tags[i])
    return ret

def strip_accents(string):
    return tokenizer.basic_tokenizer._run_strip_accents(string)


def tokenize_with_span(string):
    '''
    depends on tokenize
    '''
    tokens = tokenize(string)
    token_spans = []

    i_token = 0
    i_string = 0
    while len(token_spans) < len(tokens):
        # fix -1 at the end
        if tokens[i_token] == '[UNK]':
            token_spans.append((i_string, -1))
            i_string += 1
            i_token += 1
            continue
            
        # strip '##' to adapt BertTokenizer
        if not tokens[i_token].strip('##').startswith(string[i_string]):
            i_string += 1
            continue

        token_spans.append((i_string, i_string + len(tokens[i_token].strip('##'))))
        i_string += len(tokens[i_token].strip('##'))
        i_token += 1
    
    # fix -1 caused by [UNK]
    for i, span in enumerate(token_spans):
        if span[1] == -1:
            if i == len(token_spans)-1:
                token_spans[i] = (token_spans[i][0], len(string))
            else:
                token_spans[i] = (token_spans[i][0], token_spans[i+1][0])

    return tokens, token_spans

def get_span_dict(X):
    '''
    in: [0, 0, 0, 1, 1, 1, 0, 0, 0]
    out: {
        0: [(0, 3), (6, 9)],
        1: [(3, 6)],
    }
    '''
    span_dict = defaultdict(list)
    current = None
    for i, x in enumerate(X):
        if current == x:
            span_dict[x][-1][1] = i+1
        elif current != x:
            current = x
            span_dict[x].append([i, i+1])

    return span_dict

def tags_to_ent_tags(tags, N, M=2):
    onehot_tags = to_one_hot(tags, N)
    onehot_tags2 = onehot_tags[:, :, 1:].view(*tags.shape, (N-1)//M, M)
    onehot_tags3 = torch.cat([torch.zeros([*onehot_tags2.shape[:-1], 1])+0.1, onehot_tags2], dim=-1)
    ent_tags = onehot_tags3.argmax(-1)
    return ent_tags

def tag2span(tags, return_types=False, arbitrary_tag=True):
    '''
    IOBE
    '''
    if arbitrary_tag:
        tags = ALL2BIO(tags)
        tags = BIO2BIOE(tags)
    spans = []
    types = []
    _span = _type = None
    for i, t in enumerate(tags):
        if (t[0] == 'B' or t == 'O') and _span is not None:
            spans.append(_span)
            types.append(_type)
            _span = _type = None
        if t[0] == 'B':
            _span = [i, i+1]
            _type = t[2:]
        if t[0] == 'I':
            if _span is not None:
                _span[1] = i+1
        if t[0] == 'E':
            if _span is not None:
                _span[1] = i+1
    if _span is not None:
        spans.append(_span)
        types.append(_type)
        
    if return_types:
        return spans, types
    return spans

def combine_tags_list(tags_list):
    tags = []
    for i in range(len(tags_list[0])):
        for tag in [_tags[i] for _tags in tags_list]:
            if tag != 'O':
                tags.append(tag)
                break
        else:
            tags.append('O')
    return tags

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):

    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x