import pandas as pd
import os
import json
from tqdm import tqdm
data_dir = '/data/renma/persona_chat/datasets/ready4training/KdConv'

train = pd.read_csv(data_dir + '/train.csv')
valid = pd.read_csv(data_dir + '/valid.csv')
test = pd.read_csv(data_dir + '/test.csv')


def process(df):
    output = []
    chat_history = df['attr_sentence'] + '。' + df['chat_history']
    chat_history = chat_history.str.replace('\[SEP]', '').to_list()
    for src, tgt in tqdm(zip(chat_history, df['message'].to_list())):
        output.append([src, tgt])
    
    return output

def write_json(data_lst, dataset):
    with open(data_dir + '/' + dataset + '.json', 'w', encoding='utf-8') as f:
        for item in data_lst:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

train_lst = process(train)
valid_lst = process(valid)
test_lst = process(test)
write_json(train_lst, 'train')
write_json(valid_lst, 'valid')
test['inputs4nezha'] = test_lst
test.to_csv(data_dir + '/test4nezha.csv', index=False)


    
