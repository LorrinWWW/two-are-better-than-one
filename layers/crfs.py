
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from utils import *
from functions import *

try:
    from allennlp.modules import ConditionalRandomField
except Exception as e:
    print("We adopt CRF implemented by allennlp, please install it first.")
    raise e



'''
The two viterbi implementations does not include [START] and [END] tokens
'''
def viterbi_decode(score, transition_params):
    trellis = np.zeros_like(score) # (L, K)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]
    
    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score

def viterbi_decode_torch(score, transition_params, mask=None):
    trellis = torch.zeros_like(score) # (B, L, K)
    backpointers = torch.zeros_like(score, dtype=torch.int32) # (B, L, K)
    trellis[:, 0] = score[:, 0]
    
    for t in range(1, score.shape[1]):
        v = trellis[:, t - 1, :, None] + transition_params
        tmp0, tmp1 = torch.max(v, 1)
        trellis[:, t] = score[:, t] + tmp0
        backpointers[:, t] = tmp1

    trellis = trellis.cpu().detach().numpy()
    backpointers = backpointers.cpu().detach().numpy()

    viterbi_list = []
    viterbi_score_list = []
    for i in range(backpointers.shape[0]):
        viterbi = [np.argmax(trellis[i, -1])]
        for bp in reversed(backpointers[i, 1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = np.max(trellis[i, -1], -1)
        viterbi_list.append(viterbi)
        viterbi_score_list.append(viterbi_score)
    return viterbi_list, viterbi_score_list


class CRF(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.crf = ConditionalRandomField(
            num_tags=config.tag_vocab_size, 
            include_start_end_transitions=False,
        )
    
    def forward(self, 
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor = None,
                reduction: str = 'sum'):
        if mask is None:
            mask = torch.ones(*inputs.size()[:2], dtype=torch.long).to(inputs.device)
        
        log_denominator = self.crf._input_likelihood(inputs, mask)
        log_numerator = self.crf._joint_likelihood(inputs, tags, mask)
        loglik = log_numerator - log_denominator
        
        if reduction == 'sum':
            loglik = loglik.sum()
        elif reduction == 'mean':
            loglik = loglik.mean()
        elif reduction == 'none':
            pass
        return loglik
    
    def decode(self, inputs, mask=None):
        if mask is None:
            mask = torch.ones(*inputs.shape[:2], dtype=torch.long).to(inputs.device)
#         preds = self.crf.viterbi_tags(inputs, mask)
#         preds, scores = zip(*preds)
        preds, scores = viterbi_decode_torch(inputs, self.crf.transitions)
        return list(preds)
    

        