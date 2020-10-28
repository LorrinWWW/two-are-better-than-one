import json


for dataset_type in ['train', 'dev', 'test']:
    with open(f'./json/{dataset_type}.json') as f:
        exec(f'{dataset_type} = []')
        dataset = eval(dataset_type)
        for line in f:
            doc = json.loads(line)
            dataset.append(doc)
        
        
for dataset_type in ['train', 'dev', 'test']:
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

flag = 'ACE05'
for dataset_type in ['train', 'dev', 'test']:
    dataset_fine = eval(f'{dataset_type}_fine')
    
    if dataset_type == 'dev':
        dataset_type = 'valid'
    
    with open(f'../unified/{dataset_type}.{flag}.json', 'w') as f:
        json.dump(dataset_fine, f)