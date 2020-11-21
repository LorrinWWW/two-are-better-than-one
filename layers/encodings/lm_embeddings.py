
import os
import math
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import flair
from flair.embeddings import *
from flair.data import *


from transformers import *

from utils import *
from functions import *

from layers.indexings import *


class BertEmbeddingsWithHeads(BertEmbeddings):
    def __init__(
        self,
        bert_model_or_path: str = "bert-base-uncased",
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
        super().__init__()

        if "distilbert" in bert_model_or_path:
            try:
                from transformers import DistilBertTokenizer, DistilBertModel
            except ImportError:
                log.warning("-" * 100)
                log.warning(
                    "ATTENTION! To use DistilBert, please first install a recent version of transformers!"
                )
                log.warning("-" * 100)
                pass

            self.tokenizer = DistilBertTokenizer.from_pretrained(bert_model_or_path)
            self.model = DistilBertModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path,
                output_hidden_states=True,
                output_attentions=True,
            )
        elif "albert" in bert_model_or_path:
            self.tokenizer = AlbertTokenizer.from_pretrained(bert_model_or_path)
            self.model = AlbertModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path,
                output_hidden_states=True,
                output_attentions=True,
            )
        elif "roberta" in bert_model_or_path:
            self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_or_path)
            self.model = RobertaModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path,
                output_hidden_states=True,
                output_attentions=True,
            )
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)
            self.model = BertModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path,
                output_hidden_states=True,
                output_attentions=True,
            )
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.name = str(bert_model_or_path)
        self.static_embeddings = True


    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""

        # first, find longest sentence in batch
        longest_sentence_in_batch: int = len(
            max(
                [
                    self.tokenizer.tokenize(sentence.to_tokenized_string())
                    for sentence in sentences
                ],
                key=len,
            )
        )

        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch
        )
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(
            flair.device
        )
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(
            flair.device
        )

        # put encoded batch through BERT model to get all hidden states of all encoder layers
        self.model.to(flair.device)
        self.model.eval()
        tmp = self.model(all_input_ids, attention_mask=all_input_masks)
        all_encoder_layers = tmp[-2]
        all_attentions = tmp[-1]

        with torch.no_grad():
            
            all_attentions = torch.cat(all_attentions, 1)

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        if self.use_scalar_mix:
                            layer_output = all_encoder_layers[int(layer_index)][
                                sentence_index
                            ]
                        else:
                            layer_output = all_encoder_layers[int(layer_index)][
                                sentence_index
                            ]
                        all_layers.append(layer_output[token_index])

                    if self.use_scalar_mix:
                        sm = ScalarMix(mixture_size=len(all_layers))
                        sm_embeddings = sm(all_layers)
                        all_layers = [sm_embeddings]

                    subtoken_embeddings.append(torch.cat(all_layers))

                # get the current sentence object
                token_idx = 0
                for token in sentence:
                    # add concatenated embedding to sentence
                    token_idx += 1

                    if self.pooling_operation == "first":
                        # use first subword embedding if pooling operation is 'first'
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # otherwise, do a mean over all subwords in token
                        embeddings = subtoken_embeddings[
                            token_idx : token_idx
                            + feature.token_subtoken_count[token.idx]
                        ]
                        embeddings = [
                            embedding.unsqueeze(0) for embedding in embeddings
                        ]
                        try:
                            mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        except Exception as e:
                            print(sentence)
                            print(feature.tokens)
                            print(token)
                            print(token_idx, feature.token_subtoken_count[token.idx])
                            print(embeddings)
                            print(len(subtoken_embeddings))
                            raise e
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1
                    
                #### attentions
                
                a = all_attentions[sentence_index] # (L*H, T, T)
                
                n_tokens = len(sentence.tokens)
                n_subtokens = a.size(-1)
                d_heads = a.size(0)
                
                b = torch.zeros([d_heads, n_tokens, n_subtokens], dtype=a.dtype, device=a.device)
                c = torch.zeros([d_heads, n_tokens, n_tokens], dtype=a.dtype, device=a.device)
                
                bias = 1
                for k in range(1, n_tokens+1):
                    v = feature.token_subtoken_count[k]
                    b[:, k-1] = a[:, bias:bias+v].sum(1) # mean of max
                    bias += v
                bias = 1
                for k in range(1, n_tokens+1):
                    v = feature.token_subtoken_count[k]
                    c[:, :, k-1] = b[:, :, bias:bias+v].sum(2) # sum if attention probs; mean if attention scores
                    bias += v
                
                sentence.set_embedding('heads', c.permute(1,2,0))
                
        return sentences
    
    

def form_sentence(tokens):
    s = Sentence()
    for w in tokens:
        if w == '´' or w == '˚':
            s.add_token(Token('-'))
        else:
            s.add_token(Token(w))
    return s

def get_embs(s):
    ret = []
    for t in s:
        ret.append(t.get_embedding().cpu().numpy())
    return np.stack(ret, axis=0)


class PreEmbeddedLM(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config 
        self.device = config.device
        self.lm_emb_path = config.lm_emb_path
        
        self.lm_emb_is_file = os.path.isfile(self.lm_emb_path)
        
        if self.lm_emb_is_file:
            with open(self.lm_emb_path, 'rb') as f:
                self.emb_dict = pickle.load(f)
        elif 'bert' in self.lm_emb_path:
            print(f'{self.lm_emb_path} is not file, try load as bert model.')
            self.lm = [BertEmbeddingsWithHeads(
                self.lm_emb_path, layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean")]
            print(f'{self.lm_emb_path} loaded successfully.')
            print('Note it only supports default options now, i.e.: ')
            print('''  layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean"''')
            
            
    def forward(self, batch_tokens):
        
        tmp = self.emb_tokens(batch_tokens)
        if isinstance(tmp[0], list) or isinstance(tmp[0], tuple):
            embs = [x[0] for x in tmp]
            heads = [x[1] for x in tmp]
        else:
            embs = tmp
            heads = None
            
        embs = pad_sequences(embs, maxlen=None, dtype='float32',
                  padding='post', truncating='post', value=0.)
        embs = torch.from_numpy(embs).float()
        embs = embs.to(self.device)
        
        if heads is not None:
            heads = pad_matrices(heads, maxlen=None, dtype='float32',
                  padding='post', truncating='post', value=0.)
            heads = torch.from_numpy(heads).float()
            
            heads = heads.to(self.device)
        
        return (embs, heads)
    
    def emb_tokens(self, tokens_list):
        
        rets = []
        
        if self.lm_emb_is_file:
        
            for tokens in tokens_list:
                tokens = tuple(tokens)
                if tokens not in self.emb_dict:
                    raise Exception(f'{tokens} not pre-emb')
                
                rets.append(self.emb_dict[tokens])
        
        else:
            
            s = [form_sentence(tuple(tokens)) for tokens in tokens_list]
            try:
                self.lm[0].embed(s)
            except Exception as e:
                for _s in s:
                    print(_s)
                raise e
                
            rets = []
            for _s in s:
                emb = get_embs(_s)
                if self.config.head_emb_dim > 0:
                    heads = _s.get_embedding().cpu().numpy()
                    rets.append((emb, heads))
                else:
                    rets.append(emb)
            
        return rets

