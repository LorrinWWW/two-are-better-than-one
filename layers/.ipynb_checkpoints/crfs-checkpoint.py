
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

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

def viterbi_decode_torch(score, transition_params):
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
    
class DTCRF(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tag_form = config.tag_form
        
        self.crf = ConditionalRandomField(
            num_tags=config.tag_vocab_size, 
            include_start_end_transitions=False,
        )
        del self.crf.transitions # must del parameter before assigning a tensor
        self.crf.transitions = None
        del self.crf._constraint_mask
        num_tags = config.tag_vocab_size
        constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(1.).to(config.device)
        self.crf._constraint_mask = constraint_mask #torch.nn.Parameter(constraint_mask, requires_grad=False)
        
        if self.tag_form == 'iobes':
            M = 4
        elif self.tag_form == 'iob2':
            M = 2
        else:
            raise Exception(f'unsupported tag form: {self.tag_form}')
        
        N = config.tag_vocab_size
        E = (config.tag_vocab_size - 1) // M
            
        self.N, self.M, self.E = N, M, E
        self.p_in = nn.Parameter(torch.randn([M, M], dtype=torch.float32))
        self.p_cross = nn.Parameter(torch.randn([M, M], dtype=torch.float32))
        self.p_out = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.p_to_out = nn.Parameter(torch.randn(M, dtype=torch.float32))
        self.p_from_out = nn.Parameter(torch.randn(M, dtype=torch.float32))
        
        self.need_update = True
        
    def p_to_cpu(self):
        if self.p_in.device.type != 'cpu':
            self.p_in.data = self.p_in.data.cpu()
            self.p_cross.data = self.p_cross.data.cpu()
            self.p_out.data = self.p_out.data.cpu()
            self.p_to_out.data = self.p_to_out.data.cpu()
            self.p_from_out.data = self.p_from_out.data.cpu()
            
    def update_transitions(self):
        ### build transition matrix (operation on cpu)
        M, N, E = self.M, self.N, self.E
        extended = torch.zeros([N, N])#.to(self.config.device) # extended transition matrix
        extended[0, 0] = self.p_out # O to O
        for e in range(E):
            extended[0, e*M+1: e*M+1+M] = self.p_from_out
            extended[e*M+1: e*M+1+M, 0] = self.p_to_out
            
        for e0 in range(E):
            extended[e0*M+1: e0*M+1+M, e0*M+1: e0*M+1+M] = self.p_in
            for e1 in range(e0+1, E):
                extended[e0*M+1: e0*M+1+M, e1*M+1: e1*M+1+M] = self.p_cross
                extended[e1*M+1: e1*M+1+M, e0*M+1: e0*M+1+M] = self.p_cross
        self.crf.transitions = extended.to(self.config.device)
        ### finish building transition matrix
            
    def forward(self, inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor = None,
                reduction: str = 'sum'):
        
        self.p_to_cpu()
        self.update_transitions()
        self.need_update = True
        
        if mask is None:
            mask = torch.ones(*inputs.size()[:2], dtype=torch.long).to(inputs.device)
        
        log_denominator = self.crf._input_likelihood(inputs, mask)
        log_numerator = self.crf._joint_likelihood(inputs, tags, mask)
        loglik = log_numerator - log_denominator
        
        if reduction == 'sum':
            loglik = loglik.sum()
        elif reduction == 'mean':
            loglik = loglik.mean()
        elif reduction == 'token_mean':
            loglik = loglik.mean()
        elif reduction == 'none':
            pass
        
        return loglik
    
    def decode(self, inputs, mask=None):

        if self.need_update:
            self.update_transitions()
            self.need_update = False
            
        if mask is None:
            mask = torch.ones(*inputs.shape[:2], dtype=torch.long).to(inputs.device)
            
#         preds = self.crf.viterbi_tags(inputs, mask)
#         preds, scores = zip(*preds)
        preds, scores = viterbi_decode_torch(inputs, self.crf.transitions)
    
        return list(preds)

class DCCRF(nn.Module):

    def __init__(self, config, input_dim=None):
        super().__init__()
        self.config = config
        self.tag_form = config.tag_form
        tag_form = config.tag_form
        
        self.crf = ConditionalRandomField(
            num_tags=config.tag_vocab_size, 
            include_start_end_transitions=False,
        )
        del self.crf.transitions # must del parameter before assigning a tensor
        self.crf.transitions = None
        del self.crf._constraint_mask
        num_tags = config.tag_vocab_size
        constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(1.).to(config.device)
        self.crf._constraint_mask = constraint_mask
        
        if tag_form == 'iobes':
            M = 4
        elif tag_form == 'iob2':
            M = 2
        else:
            raise Exception(f'unsupported tag form: {tag_form}')
        
        N = config.tag_vocab_size
        E = (config.tag_vocab_size - 1) // M
        A = 4
            
        self.N, self.M, self.E, self.A = N, M, E, A
        self.p_in = nn.Parameter(torch.randn([A, M, M], dtype=torch.float32))
        self.p_cross = nn.Parameter(torch.randn([M, M], dtype=torch.float32))
        self.p_out = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.p_to_out = nn.Parameter(torch.randn([M], dtype=torch.float32))
        self.p_from_out = nn.Parameter(torch.randn([M], dtype=torch.float32))
        
        if input_dim is None:
            input_dim = config.hidden_dim
        self.block_attn = nn.Linear(input_dim, A)
        init_linear(self.block_attn)
        self.dropout = nn.Dropout(0.5)
        
    def p_to_cpu(self):
        if self.p_out.device.type != 'cpu':
#             self.p_in.data = self.p_in.data.cpu()
            self.p_cross.data = self.p_cross.data.cpu()
            self.p_out.data = self.p_out.data.cpu()
            self.p_to_out.data = self.p_to_out.data.cpu()
            self.p_from_out.data = self.p_from_out.data.cpu()
            
    def update_transitions(self, hiddens, entity_mask=None):
        ### build transition matrix (operation on cpu)
        M, N, K, A = self.M, self.N, self.K, self.A
        
        ### predict block
        block_atten = self.block_attn(hiddens)
        if entity_mask is not None:
            block_atten -= 999*(1.-entity_mask)[:, None]
        block_atten = F.softmax(block_atten, 0)
        block_atten = F.softmax(block_atten*10, -1)
        
        p_in = (self.p_in[None] * block_atten[:, :, None, None]).mean(1).cpu() # (K, A, N, N) => (K, N, N)

        ### build extended
        extended = torch.zeros([N, N]) #.to(self.config.device) # extended transition matrix
        extended[0, 0] = self.p_out # O to O
        
        for e in range(E):
            extended[0, e*M+1: e*M+1+M] = self.p_from_out
            extended[e*M+1: e*M+1+M, 0] = self.p_to_out
            
        for e0 in range(E):
            extended[e0*M+1: e0*M+1+M, e0*M+1: e0*M+1+M] = p_in[k0]
            for e1 in range(e0+1, E):
                extended[e0*M+1: e0*M+1+M, e1*M+1: e1*M+1+M] = self.p_cross
                extended[e1*M+1: e1*M+1+M, e0*M+1: e0*M+1+M] = self.p_cross
        self.crf.transitions = extended.to(self.config.device)
        ### finish building transition matrix
            
    def forward(self, 
                inputs: torch.Tensor,
                tags: torch.Tensor,
                hiddens: torch.Tensor,
                mask: torch.ByteTensor = None,
                entity_mask = None,
                reduction: str = 'sum'):
        
        self.p_to_cpu()
        self.update_transitions(hiddens, entity_mask)
        
        if mask is None:
            mask = torch.ones(*inputs.size()[:2], dtype=torch.long).to(inputs.device)
        
        log_denominator = self.crf._input_likelihood(inputs, mask)
        log_numerator = self.crf._joint_likelihood(inputs, tags, mask)
        loglik = log_numerator - log_denominator
        
        if reduction == 'sum':
            loglik = loglik.sum()
        elif reduction == 'mean':
            loglik = loglik.mean()
        elif reduction == 'token_mean':
            loglik = loglik.mean()
        elif reduction == 'none':
            pass
        
        return loglik
    
    def decode(self, inputs, hiddens, mask=None, entity_mask=None):

        self.update_transitions(hiddens, entity_mask)
            
        if mask is None:
            mask = torch.ones(*inputs.shape[:2], dtype=torch.long).to(inputs.device)
            
#         preds = self.crf.viterbi_tags(inputs, mask)
#         preds, scores = zip(*preds)
        preds, scores = viterbi_decode_torch(inputs, self.crf.transitions)
    
        return list(preds)