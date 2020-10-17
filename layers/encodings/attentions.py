
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
        
        
class TransformerEncoding(nn.Module):
    
    def __init__(self, config, nhead=4, num_layers=2, norm_output=True):
        super().__init__()
        self.config = config
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_dim, nhead=nhead)
        if norm_output:
            norm = nn.LayerNorm(self.config.hidden_dim)
        else:
            norm = None
        self.attn = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm)
        
    def forward(self, inputs, mask=None, attn_mask=None):
        inputs = inputs.permute(1,0,2)
        src_key_padding_mask = None if mask is None else ~mask
        outputs = self.attn(inputs, src_key_padding_mask=src_key_padding_mask, mask=attn_mask)
        outputs = outputs.permute(1,0,2)
        return outputs
    
class TransformerDecoding(nn.Module):
    
    def __init__(self, config, nhead=4, num_layers=2, norm_output=True):
        super().__init__()
        self.config = config
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.hidden_dim, nhead=nhead)
        if norm_output:
            norm = nn.LayerNorm(self.config.hidden_dim)
        else:
            norm = None
        self.attn = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers, norm=norm)
        
    def forward(self, tgt_inputs, memory_inputs, tgt_mask=None, memory_mask=None):
        tgt_inputs = tgt_inputs.permute(1,0,2)
        memory_inputs = memory_inputs.permute(1,0,2)
        tgt_key_padding_mask = None if tgt_mask is None else ~tgt_mask
        memory_key_padding_mask = None if memory_mask is None else ~memory_mask
        outputs = self.attn(
            tgt_inputs, memory_inputs,
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=memory_key_padding_mask)
        outputs = outputs.permute(1,0,2)
        return outputs

    
class AttentionEncoding(nn.Module):
    ''' n to 1 '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = nn.Linear(config.hidden_dim, 1)
        init_linear(self.attention)
        
    def forward(self, inputs, mask=None):
        a = self.attention(inputs) # (B, T, H) => (B, T, 1)
        if mask is not None:
            a -= 999*(~mask).float()[:, :, None]
        a = F.softmax(a, dim=1) # (B, T, 1)
        outputs = (a*inputs).sum(1) # (B, H)
        return outputs
    
class CustomMultiheadAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tmp = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2, heads = tmp[0], tmp[1]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, heads

    
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)