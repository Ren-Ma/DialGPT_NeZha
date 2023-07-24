#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2020/12/23 5:15 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

import os
import sys
from tqdm import tqdm
sys.path.append('.')
import keras
from src.TextGen.training.nezha_gpt_train import model, tokenizer
from src.TextGen.bert_layers.snippets import DataGenerator
from src.TextGen.bert_layers.snippets import sequence_padding

file_root = os.path.dirname(__file__)

maxlen = 512
# 可以更改参数
batch_size = 48
steps_per_epoch = 10000
epochs = 2

def process_data(timesince='2020_12_22', data_dir=os.path.join(file_root,'../../TextBro/data')):
    pass

def corpus(data_dir):
    """循环读取语料
    """
    while True:
        data_all = []
        for file in tqdm(os.listdir(data_dir)):
            if not file.endswith('.txt'):
                continue
            with open(os.path.join(data_dir,file),'r') as f:
                data = f.read().split('\n\n')
                data_all.extend(data)
        print('data length {}'.format(len(data_all)))
        for l in data_all:
            l = l.split('\n')
            yield l

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, texts in self.sample(random):
            token_ids, segment_ids = [tokenizer._token_start_id], [0]
            for i, text in enumerate(texts):
                ids = tokenizer.encode(text)[0][1:]
                if len(token_ids) + len(ids) <= maxlen:
                    token_ids.extend(ids)
                    segment_ids.extend([i % 2] * len(ids))
                else:
                    break
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class Evaluator(keras.callbacks.Callback):
    """保存模型权重
    """
    def on_epoch_end(self, epoch, logs=None):
        while True:
            try:
                model.save_weights(os.path.join(file_root,'latest_model_dynamic.weights'))
                break
            except:
                print(u'保存失败，正在重试...')

def run():
    tag = 'train'
    if tag == 'train':
        data_dir = os.path.join(file_root,'../GPT2_chitchat/data/process_data')
        evaluator = Evaluator()
        train_generator = data_generator(corpus(data_dir), batch_size)
        # 加载最后模型权重
        model.load_weights(os.path.join(file_root, 'latest_model.weights'))
        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[evaluator]
        )
    else:
        model.load_weights(os.path.join(file_root,'latest_model_dynamic.weights'))