#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2020/11/12 1:46 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

#! -*- coding: utf-8 -*-
# NEZHA模型做闲聊任务
# 训练脚本
# 训练环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.8.4
import sys
import os
import math
import pandas as pd
sys.path.append('.')

import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from src.TextGen.bert_layers.backend import keras, K
from src.TextGen.bert_layers.layers import Loss
from src.TextGen.bert_layers.models import build_transformer_model
from src.TextGen.bert_layers.tokenizers import Tokenizer, load_vocab
from src.TextGen.bert_layers.optimizers import Adam
from src.TextGen.bert_layers.optimizers import extend_with_weight_decay
from src.TextGen.bert_layers.optimizers import extend_with_gradient_accumulation
from src.TextGen.bert_layers.snippets import sequence_padding, open
from src.TextGen.bert_layers.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model

file_root = os.path.dirname(__file__)

maxlen = 512
batch_size = 28
epochs = 3

# nezha配置
config_path = os.path.join('/data/PLM/nezha/nezha_gpt_dialog/config.json')
checkpoint_path = os.path.join('/data/PLM/nezha/nezha_gpt_dialog/model.ckpt')
dict_path = os.path.join('/data/PLM/nezha/nezha_gpt_dialog/vocab.txt')


# def corpus(data_dir):
#     """循环读取语料
#     """
#     while True:
#         data_all = []
#         for file in tqdm(os.listdir(data_dir)):
#             if not file.endswith('.txt'):
#                 continue
#             with open(os.path.join(data_dir,file),'r') as f:
#                 data = f.read().split('\n\n')
#                 data_all.extend(data)
#         print('data length {}'.format(len(data_all)))
#         for l in data_all:
#             l = l.split('\n')
#             yield l

# def corpus():
#     """循环读取语料
#     """
#     while True:
#         with open(os.path.join(file_root,'nezha_gpt_dialog/LCCD-large-shuf.json')) as f:
#             for l in f:
#                 l = json.loads(l)
#                 yield l

def kdconv_read(data_json):
    """循环读取语料
    """
    while True:
        with open(data_json) as f:
            for l in f:
                l = json.loads(l)
                yield l

# 加载并精简词表
token_dict, keep_tokens = load_vocab(dict_path=dict_path, 
                                    simplified=True,
                                    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'])

# 补充词表
compound_tokens = []
for l in open(os.path.join(file_root,'nezha_gpt_dialog/user_tokens.csv'), encoding='utf-8'):
    token, count = l.strip().split('\t')
    if int(count) >= 10 and token not in token_dict:
        token_dict[token] = len(token_dict)
        compound_tokens.append([0])

# 建立分词器
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, texts in self.sample(random):
            token_ids, segment_ids = [tokenizer._token_start_id], [0]
            for i, text in enumerate(texts):
                ids = tokenizer.encode(text)[0][1:]  # 把句首的[CLS]去掉，句尾的[SEP]留着
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

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    def compute_loss(self, inputs, mask=None):
        # y_true, y_pred = inputs
        y_true, segment_ids, y_pred = inputs
        print("inputs是：", inputs)
        print("y_true是：", y_true) # Tensor("Input-Token:0", shape=(?, ?), dtype=float32)
        print("y_pred是：", y_pred) # Tensor("MLM-Activation/truediv:0", shape=(?, ?, 14194), dtype=float32)
        print("mask是：", mask) # [None, <tf.Tensor 'Transformer-11-FeedForward-Add/All:0' shape=(?, ?) dtype=bool>]
        # y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        # 当采用UniLM时，用segment_ids作为mask（输入都可见为1，输出都为0）@ ren ma 2022.11.17
        print("segment_ids是", segment_ids)
        y_mask = K.cast(segment_ids[:, 1:], dtype=K.floatx())
        print("y_mask是：", y_mask) # Tensor("cross_entropy_1/strided_slice:0", shape=(?, ?), dtype=float32)
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask) # Tensor("cross_entropy_1/truediv:0", shape=(), dtype=float32)
        return loss


model = build_transformer_model(config_path, checkpoint_path, model='nezha',
                                # application='lm',
                                application='unilm', # @ ren ma 2022.11.17
                                keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
                                compound_tokens=compound_tokens)  # 要扩充的词表

output = CrossEntropy(2)  # 进入layers/Loss.__init__() @ ren ma 2022.11.15
# output = output([model.inputs[0], model.outputs[0]])  # 进去base_layer/Layer.__call__() @ ren ma 2022.11.15
output = output([model.inputs[0], model.inputs[1], model.outputs[0]])  # 把segment_ids也加入 @ ren ma 2022.11.17

model = Model(model.inputs, output)  # network/Network.__init__() @ ren ma 2022.11.17
model.summary()

AdamW = extend_with_weight_decay(Adam, 'AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
optimizer = AdamWG(learning_rate=2e-5,
                    weight_decay_rate=0.01, 
                    exclude_from_weight_decay=['Norm', 'bias'],
                    grad_accum_steps=16)
model.compile(optimizer=optimizer)

class Evaluator(keras.callbacks.Callback):
    """保存模型权重
    """
    def on_epoch_end(self, epoch, logs=None):
        while True:
            try:
                model.save_weights(os.path.join(file_root,'kdconv_weights1118.weights'))
                break
            except:
                print(u'保存失败，正在重试...')

if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(True)
    # data_dir = os.path.join(file_root,'../GPT2_chitchat/data/process_data')
    evaluator = Evaluator()
    # train_generator = data_generator(corpus(data_dir), batch_size)
    # train_generator = data_generator(corpus(), batch_size)
    data_dir = '/data/renma/persona_chat/datasets/ready4training/KdConv'
    train_json = data_dir + '/train.json'
    valid_json = data_dir + '/valid.json'
    train_generator = data_generator(kdconv_read(train_json), batch_size)
    valid_generator = data_generator(kdconv_read(valid_json), batch_size)
    # fit_generator里不能传入batch_size，要用steps_per_epoch @ ren ma 2022.11.18
    # 每个step跑一个batch，过steps_per_epoch后跑完一个epoch 
    steps_per_epoch = math.ceil(len(pd.read_json(train_json, lines=True)) / batch_size)
    validation_steps = math.ceil(len(pd.read_json(valid_json, lines=True)) / batch_size)

    # 加载最后模型权重
    model.load_weights(os.path.join(file_root, 'kdconv_weights1118.weights'))
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model.fit_generator(
            train_generator.forfit(),
            validation_data=valid_generator.forfit(),
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[evaluator])  # 进去 @ ren ma 2022.11.15


