import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import json
import math
import pickle
from tqdm import tqdm

from transformers import *
from typing import *

import flair
from flair.embeddings import *
from flair.data import *

from flair.embeddings import _build_token_subwords_mapping_gpt2, _build_token_subwords_mapping, _extract_embeddings


parser = argparse.ArgumentParser(description='Arguments for training.')

parser.add_argument('--dataset',
                    default='ACE05',
                    action='store',)

parser.add_argument('--model_name',
                    default='roberta-large',
                    action='store',)

parser.add_argument('--save_path',
                    default='./wv/lm.emb.pkl',
                    action='store',)

parser.add_argument('--save_attention',
                    default=1,
                    action='store',)

parser.add_argument('--layers',
                    default='mean',
                    action='store',)

args = parser.parse_args()


if 'roberta' in args.model_name:
    _Model = RobertaModel
    _ClassEmbeddings = RoBERTaEmbeddings
    _Tokenizer = RobertaTokenizer
else:
    raise Exception('unsupported ckpt!')
    
# elif 'albert' in args.model_name:
#     _Model = AlbertModel
#     _ClassEmbeddings = BertEmbeddings
#     _Tokenizer = AlbertTokenizer
# else:
#     _Model = BertModel
#     _ClassEmbeddings = BertEmbeddings
#     _Tokenizer = BertTokenizer


    
class BertEmbeddingsWithHeads(_ClassEmbeddings):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
        use_scalar_mix: bool = False,
    ):
        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file
        and vocab file (names of three files should be - config.json, pytorch_model.bin/model.chkpt, vocab.txt)
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        super(_ClassEmbeddings, self).__init__()
        
        self.tokenizer = _Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = _Model.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            output_hidden_states=True,
            output_attentions=True,  ## important
        )
        self.name = pretrained_model_name_or_path
        self.layers: List[int] = [int(layer) for layer in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True   ## important

        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )


    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        self.model.to(flair.device)
        self.model.eval()

        sentences = _get_transformer_sentence_embeddings(
            sentences=sentences,
            tokenizer=self.tokenizer,
            model=self.model,
            name=self.name,
            layers=self.layers,
            pooling_operation=self.pooling_operation,
            use_scalar_mix=self.use_scalar_mix,
            bos_token="<s>",
            eos_token="</s>",
        )

        return sentences

def _get_transformer_sentence_embeddings(
    sentences: List[Sentence],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    name: str,
    layers: List[int],
    pooling_operation: str,
    use_scalar_mix: bool,
    bos_token: str = None,
    eos_token: str = None,
) -> List[Sentence]:
    """
    Builds sentence embeddings for Transformer-based architectures.
    :param sentences: input sentences
    :param tokenizer: tokenization object
    :param model: model object
    :param name: name of the Transformer-based model
    :param layers: list of layers
    :param pooling_operation: defines pooling operation for subword extraction
    :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
    :param bos_token: defines begin of sentence token (used for left padding)
    :param eos_token: defines end of sentence token (used for right padding)
    :return: list of sentences (each token of a sentence is now embedded)
    """
    with torch.no_grad():
        for sentence in sentences:
            token_subwords_mapping: Dict[int, int] = {}

            if ("gpt2" in name or "roberta" in name) and "xlm" not in name:
                (
                    token_subwords_mapping,
                    tokenized_string,
                ) = _build_token_subwords_mapping_gpt2(
                    sentence=sentence, tokenizer=tokenizer
                )
            else:
                (
                    token_subwords_mapping,
                    tokenized_string,
                ) = _build_token_subwords_mapping(
                    sentence=sentence, tokenizer=tokenizer
                )

            subwords = tokenizer.tokenize(tokenized_string)

            offset = 0

            if bos_token:
                subwords = [bos_token] + subwords
                offset = 1

            if eos_token:
                subwords = subwords + [eos_token]

            indexed_tokens = tokenizer.convert_tokens_to_ids(subwords)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(flair.device)

            ############## return hidden_states and attentions
            tmp = model(tokens_tensor)
            hidden_states = tmp[-2]
            attentions = tmp[-1]
            attentions = torch.cat(attentions, dim=1).squeeze(0)

            for token in sentence.tokens:
                len_subwords = token_subwords_mapping[token.idx]

                subtoken_embeddings = _extract_embeddings(
                    hidden_states=hidden_states,
                    layers=layers,
                    pooling_operation=pooling_operation,
                    subword_start_idx=offset,
                    subword_end_idx=offset + len_subwords,
                    use_scalar_mix=use_scalar_mix,
                )

                offset += len_subwords

                final_subtoken_embedding = torch.cat(subtoken_embeddings)
                token.set_embedding(name, final_subtoken_embedding)
                
            ################# added
            a = attentions # (L*H, T, T)
                
            n_tokens = len(sentence.tokens)
            n_subtokens = a.size(-1)
            d_heads = a.size(0)
                
            b = torch.zeros([d_heads, n_tokens, n_subtokens], dtype=a.dtype, device=a.device)
            c = torch.zeros([d_heads, n_tokens, n_tokens], dtype=a.dtype, device=a.device)
                
            bias = 1
            for k in range(1, n_tokens+1):
                v = token_subwords_mapping[k]
                b[:, k-1] = a[:, bias:bias+v].sum(1) # mean of max
                bias += v
            bias = 1
            for k in range(1, n_tokens+1):
                v = token_subwords_mapping[k]
                c[:, :, k-1] = b[:, :, bias:bias+v].sum(2) # sum if attention probs; mean if attention scores
                bias += v
                
            sentence.set_embedding('heads', c.permute(1,2,0))

    return sentences
    
    

def form_sentence(tokens):
    s = Sentence()
    for w in tokens:
        if w == '´' or w == '˚': # avoid unsupported char
            s.add_token(Token('-'))
        else:
            s.add_token(Token(w))
    return s

def get_embs(s):
    ret = []
    for t in s:
        ret.append(t.get_embedding().cpu().numpy())
    return np.stack(ret, axis=0)


if args.layers == 'mean':
    embedding = BertEmbeddingsWithHeads(args.model_name, layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean")
else:
    embedding = BertEmbeddingsWithHeads(args.model_name, layers=args.layers, pooling_operation="mean")


flag = args.dataset
dataset = []
with open(f'./datasets/unified/train.{flag}.json') as f:
    dataset += json.load(f)
with open(f'./datasets/unified/valid.{flag}.json') as f:
    dataset += json.load(f)
with open(f'./datasets/unified/test.{flag}.json') as f:
    dataset += json.load(f)
    
    
bert_emb_dict = {}
for item in tqdm(dataset):
    tokens = tuple(item['tokens'])
    s = form_sentence(tokens)
    embedding.embed(s)
    emb = get_embs(s).astype('float16')
    heads = s.get_embedding().cpu().numpy().astype('float16')
    if int(args.save_attention):
        bert_emb_dict[tokens] = (emb, heads)
    else:
        bert_emb_dict[tokens] = emb
    
with open(args.save_path, 'wb') as f:
    pickle.dump(bert_emb_dict, f)