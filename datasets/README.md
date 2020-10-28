
# Preprocessing ACE

Scripts are adapted from [DyGIE](https://github.com/luanyi/DyGIE/tree/master/preprocessing) and [LSTM-ER](https://github.com/tticoin/LSTM-ER/tree/master/data).

## Requirements

python3
perl
nltk (for stanford pos tagger)
java (for stanford tools)
zsh
task datasets (see below)

## Links to tasks/data sets

ACE 2004 (https://catalog.ldc.upenn.edu/LDC2005T09)
ACE 2005 (https://catalog.ldc.upenn.edu/LDC2006T06)


## Usage

### download Stanford Core NLP & POS tagger

```
cd common
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
wget http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip
unzip stanford-corenlp-full-2015-04-20.zip
unzip stanford-postagger-2015-04-20.zip
cd ..
```

### copy and convert each corpus 

Please set the environment variables for the directories, or directly put the directories in the following commands beforehand.

#### ACE 2004

```
cp -r ${ACE2004_DIR}/*/english ace2004/
cd ace2004
zsh run.zsh
mkdir -p ./json/train
mkdir -p ./json/test
python ace2json.py
python unify.py
cd ..
```

#### ACE 2005

```
cp -r ${ACE2005_DIR}/*/English ace2005/
cd ace2005
zsh run.zsh
mkdir -p ./json/
python ace2json.py
python unify.py
cd ..
```