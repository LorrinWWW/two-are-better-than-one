import json
import random

for i in range(5):
    
    for dataset_type in ['train', 'dev', 'test']:
        with open(f"./json/{dataset_type if dataset_type!='dev' else 'test'}/{i}.json") as f:
            exec(f'{dataset_type} = []')
            dataset = eval(dataset_type)
            for line in f:
                doc = json.loads(line)
                dataset.append(doc)

    for dataset_type in ['train', 'dev', 'test']:
        dataset = eval(dataset_type)
        exec(f'{dataset_type}_fine = []')
        dataset_fine = eval(f'{dataset_type}_fine')

        for doc in dataset:
            sentences = doc['sentences']
            ner = doc['ner']
            relations = doc['relations']

            bias = 0

            for sentence, ne, relation in zip(sentences,ner,relations):

                tokens = []
                for token in sentence:
                    if token.lower() == '-lrb-':
                        tokens.append('(')
                    elif token.lower() == '-rrb-':
                        tokens.append(')')
                    elif token.lower() == '-lsb-':
                        tokens.append('[')
                    elif token.lower() == '-rsb-':
                        tokens.append(']')
                    else:
                        tokens.append(token)

                item = {
                    'tokens': tokens,
                    'entities': [(i_begin-bias,i_end-bias+1,e) for i_begin, i_end, e in ne],
                    'relations': [
                        (i_begin-bias,i_end-bias+1,j_begin-bias,j_end-bias+1, r) for \
                        i_begin, i_end, j_begin, j_end, r in relation
                    ] 
                }

                dataset_fine.append(item)

                bias += len(tokens)


    random.shuffle(train_fine)
    splitter = int(len(train_fine)*0.15)
    dev_fine, train_fine = train_fine[:splitter], train_fine[splitter:]
                  
    flag = f'ACE04_{i}'
    for dataset_type in ['train', 'dev', 'test']:
        dataset_fine = eval(f'{dataset_type}_fine')
                  
        print(i, dataset_type)
        for item in dataset_fine:
            if len(item['tokens']) >= 100:
                print(len(item['tokens']))
        if dataset_type == 'train' or dataset_type == 'dev':
            dataset_fine = [item for item in dataset_fine if len(item['tokens']) < 100] # remove >100 for train to avoid OOM

        if dataset_type == 'dev':
            dataset_type = 'valid'

        with open(f'../unified/{dataset_type}.{flag}.json', 'w') as f:
            json.dump(dataset_fine, f)