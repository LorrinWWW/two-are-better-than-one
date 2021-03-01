# README

Code associated with the paper **Two are Better Than One: Joint Entity and Relation Extraction with Table-Sequence Encoders**, at EMNLP 2020

## Navi

### Resources

The datasets are available in "./datasets/".
Due to copyright issue, we cannot directly release ACE datasets, and instead, their pre-processing scripts are put in "./datasets/".

The word vectoers for each dataset are included in "./wv/";

The contextualized word embeddings and attention weights are **not** included (since they are too big).
We use library "transformers" and "flair" to generate them locally. Please refer to "./gens/gen\_\*.py".

### Model Related

The model is defined in "./models/joint\_models.py".

The basic layers are defined in "./layers/".
Specifically, the MDRNN (especially GRU) is defined in "./layers/encodings/mdrnns/gru.py".

The metric related part is located in "./data/joint\_data.py".
The micro-averaged F1 is calculated in *class JointTrainer.\_get\_metrics*;
the macro-averaged F1 is calculated in *class JointTrainerMacroF1.\_get\_metrics*.


## Dependencies

- python3
- pytorch 1.4.0
- transformers 2.9.1  
- flair 0.4.5
- gpustat


## Quick Start (Training)


1. Generate contextualized embeddings (and attention weights)

    ```shell
    python gens/gen_bert.py \
        --model albert-xxlarge-v1 \
        --dataset ACE05 \
        --save_attention 1 \
        --save_path ./wv/albert.ace05_with_heads.pkl
    ```

    where
    - if bert or albert use "gens/gen\_bert.py"; if roberta use "gens/gen\_roberta.py"
    - "--model" is the checkpoint name; 
    - "--dataset" is the name of dataset for which contextualized embeddings are prepared;
    - "--save_attention" is whether to save attention weights, note enable this will make the output file to be very big.
    - "--save\_path" is the path where features are saved to.

2. train model!

    ```shell
    python -u train.py \
        --num_layers 3 \
        --batch_size 24 \
        --evaluate_interval 500 \
        --dataset ACE05 \
        --pretrained_wv ./wv/glove.6B.100d.ace05.txt  \
        --max_epoches 5000 \
        --max_steps 20000 \
        --model_class JointModel  \
        --model_write_ckpt ./ckpts/my_model \
        --crf None \
        --optimizer adam \
        --lr 0.001 \
        --tag_form iob2 \
        --cased 0 \
        --token_emb_dim 100 \
        --char_emb_dim 30 \
        --char_encoder lstm \
        --lm_emb_dim 4096 \
        --head_emb_dim 768 \
        --lm_emb_path ./wv/albert.ace05_with_heads.pkl \
        --hidden_dim 200 \
        --ner_tag_vocab_size 32 \
        --re_tag_vocab_size 64 \
        --vocab_size 15000 \
        --dropout 0.5 \
        --grad_period 1
    ```
    
    The defualt parameters above can be used to reproduce the results in the paper on ACE05.
    
## Quick Start (Inference on custom text)

Please refer to [notebook inference](inference.ipynb).
**Note: It is not rigorously tested, and thus may be buggy.** 
If you encounter any issue, don't hesitate to report it.
    
## Training Arguments

In this section, we illustrate arguments for "train.py".

Dataset ("--dataset") needs to be specified before training. The preset data sets are:

- "ACE04\_{i}" (where {i} = 0,...,4, for 5-fold cross validation)
- "ACE05"
- "CoNLL04"
- "ADE\_{i}" (where {i} = 0,...,9, for 10-fold cross validation)

For each dataset, we prepare a subset of GloVe word vectors, which should be specified in "--pretrained\_wv":

- "./wv/glove.6B.100d.ace04.txt"
- "./wv/glove.6B.100d.ace05.txt"
- "./wv/glove.6B.100d.conll04.txt"
- "./wv/glove.6B.100d.ade.txt"

Each word is mapped to a 100d vector ("--token\_emb\_dim 100").

Contextualized word embeddings (and attention weights) are recommended to used. The path of pre-calculated contextualized word embeddings (and attention weights) needs to be specified in argument "--lm\_emb\_path". 

"--lm\_emb\_dim" should be the exact emb size of vecters stored in "lm\_emb\_path", set it to 0 when you do not want use contextualized embeddings; 

"head\_emb\_dim" is the size of attention weights of language model, it should be exactly equals to (number of layers of language model \* number of heads of language model), where the language model is what stored in "lm\_emb\_path", set it to 0 when you do not want use attention weights.

"--grad\_period" is tricky, the optimizer will accumulate gradients for "grad\_period" training steps before updating the parameters. The memory is not a problem, it should be set to 1; otherwises, reduce the batch size, and try to set grad\_period to 2 or 3 to simulate big batch size.

## Possible Problems

However, if something goes wrong, we suggest to check the following items:
- "ner\_tag\_vocab\_size" should be larger than the number of entity tags (2 \* number of entity classes + 1);
- "re\_tag\_vocab\_size" should be larger than the number of relation tags (2 \* number of relation classes + 1);
- "vocab\_size" should be larger than the vocab size in "pretrained\_wv";
- "token\_emb\_dim" should be the exact emb size of vecters stored in "pretrained\_wv";
- "lm\_emb\_dim" should be the exact emb size of vecters stored in "lm\_emb\_path", set it to 0 when you do not want use contextualized embeddings;
- "head\_emb\_dim" is the size of attention weights of language model, it should be exactly equals to (number of layers of language model \* number of heads of language model), where the language model is what stored in "lm\_emb\_path", set it to 0 when you do not want use attention weights.
  

This software does not optimize the use of memory, so OOM is likely to occur if you do not use the server designed for deep learning. Normally, a GPU with *32G* is required to run the default setting. 
We give suggestions for the case of OOM:
- reduce the batch size :)
- reduce the hidden dim :)
- "grad\_period" is used to perform "Gradient Accumulation", so batch\_size \* grad\_period is the effective batch size. However, the training time will be grad\_period times longer than usual.

## Examples

ALBERT($+x^{\ell}+T^{\ell}$), 24 batch size, 3 layers, ACE05. (requires 32G memory)

```shell
python -u train.py \
    --num_layers 3 \
    --batch_size 24 \
    --evaluate_interval 500 \
    --dataset ACE05 \
    --pretrained_wv ./wv/glove.6B.100d.ace05.txt  \
    --max_epoches 5000 \
    --max_steps 20000 \
    --model_class JointModel  \
    --model_write_ckpt ./ckpts/my_model \
    --crf None \
    --optimizer adam \
    --lr 0.001 \
    --tag_form iob2 \
    --cased 0 \
    --token_emb_dim 100 \
    --char_emb_dim 30 \
    --char_encoder lstm \
    --lm_emb_dim 4096 \
    --head_emb_dim 768 \
    --lm_emb_path ./wv/albert.ace05_with_heads.pkl \
    --hidden_dim 200 \
    --ner_tag_vocab_size 32 \
    --re_tag_vocab_size 64 \
    --vocab_size 15000 \
    --dropout 0.5 \
    --grad_period 1
```

**ALBERT($\mathbf{+x^{\ell}}$),** 24 batch size, 3 layers, ACE05. (requires 32G memory)

```shell
python -u train.py \
    --num_layers 3 \
    --batch_size 24 \
    --evaluate_interval 500 \
    --dataset ACE05 \
    --pretrained_wv ./wv/glove.6B.100d.ace05.txt  \
    --max_epoches 5000 \
    --max_steps 20000 \
    --model_class JointModel  \
    --model_write_ckpt ./ckpts/my_model \
    --crf None \
    --optimizer adam \
    --lr 0.001 \
    --tag_form iob2 \
    --cased 0 \
    --token_emb_dim 100 \
    --char_emb_dim 30 \
    --char_encoder lstm \
    --lm_emb_dim 4096 \
    --head_emb_dim 0 \
    --lm_emb_path ./wv/albert.ace05.pkl \
    --hidden_dim 200 \
    --ner_tag_vocab_size 32 \
    --re_tag_vocab_size 64 \
    --vocab_size 15000 \
    --dropout 0.5 \
    --grad_period 1
```

ALBERT($+x^{\ell}+T^{\ell}$), 24 batch size, **2** layers, ACE05. (requires 24G memory)

```shell
python -u train.py \
    --num_layers 2 \
    --batch_size 24 \
    --evaluate_interval 500 \
    --dataset ACE05 \
    --pretrained_wv ./wv/glove.6B.100d.ace05.txt  \
    --max_epoches 5000 \
    --max_steps 20000 \
    --model_class JointModel  \
    --model_write_ckpt ./ckpts/my_model \
    --crf None \
    --optimizer adam \
    --lr 0.001 \
    --tag_form iob2 \
    --cased 0 \
    --token_emb_dim 100 \
    --char_emb_dim 30 \
    --char_encoder lstm \
    --lm_emb_dim 4096 \
    --head_emb_dim 768 \
    --lm_emb_path ./wv/albert.ace05_with_heads.pkl \
    --hidden_dim 200 \
    --ner_tag_vocab_size 32 \
    --re_tag_vocab_size 64 \
    --vocab_size 15000 \
    --dropout 0.5 \
    --grad_period 1
```

ALBERT($+x^{\ell}+T^{\ell}$), (**12 batch size \* 2 grad period = 24 effective batch size**), **2** layers, ACE05. (requires 11G memory)

```shell
python -u train.py \
    --num_layers 2 \
    --batch_size 12 \
    --evaluate_interval 1000 \
    --dataset ACE05 \
    --pretrained_wv ./wv/glove.6B.100d.ace05.txt  \
    --max_epoches 5000 \
    --max_steps 20000 \
    --model_class JointModel  \
    --model_write_ckpt ./ckpts/my_model \
    --crf None \
    --optimizer adam \
    --lr 0.001 \
    --tag_form iob2 \
    --cased 0 \
    --token_emb_dim 100 \
    --char_emb_dim 30 \
    --char_encoder lstm \
    --lm_emb_dim 4096 \
    --head_emb_dim 768 \
    --lm_emb_path ./wv/albert.ace05_with_heads.pkl \
    --hidden_dim 200 \
    --ner_tag_vocab_size 32 \
    --re_tag_vocab_size 64 \
    --vocab_size 15000 \
    --dropout 0.5 \
    --grad_period 2
```




