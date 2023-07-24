#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2022/11/21 10:34
@Author  : ren ma
@Email   : ren.ma@unidt.com
"""

#! -*- coding: utf-8 -*-
# NEZHA模型做闲聊任务
# 测试脚本
# 测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.8.4
import random
import sys
import time
import ast
import jieba
import tensorflow as tf
from tqdm import tqdm
sys.path.append('.')
import numpy as np
from src.TextGen.bert_layers.backend import keras, K
from src.TextGen.bert_layers.models import build_transformer_model
from src.TextGen.bert_layers.tokenizers import Tokenizer, load_vocab
from src.TextGen.bert_layers.snippets import AutoRegressiveDecoder, sequence_padding
import os
from collections import defaultdict
import pandas as pd

file_root = os.path.dirname(__file__)

# nezha配置
config_path = os.path.join('/data/PLM/nezha/nezha_gpt_dialog/config.json')
checkpoint_path = os.path.join('/data/PLM/nezha/nezha_gpt_dialog/model.ckpt')
dict_path = os.path.join('/data/PLM/nezha/nezha_gpt_dialog/vocab.txt')

# 加载并精简词表
token_dict, keep_tokens = load_vocab(dict_path=dict_path, simplified=True, 
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

# 建立并加载模型
model = build_transformer_model(config_path, checkpoint_path, model='nezha', 
                                # application='lm', 
                                application='unilm', 
                                keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
                                compound_tokens=compound_tokens)  # 要扩充的词表
model.summary()
model.load_weights(os.path.join(file_root,'kdconv_weights1118.weights'))


class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):  # 在self.random_sample中被调用
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[:, -1]  # 返回最后一个token

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict_batch(self, inputs_batch, output_ids, states):
        token_ids_batch, segment_ids_batch = inputs_batch
        token_ids_batch = np.concatenate([token_ids_batch, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids_batch[0, -1]
        segment_ids_batch = np.concatenate([segment_ids_batch, curr_segment_ids], 1)
        return model.predict_on_batch([token_ids_batch, segment_ids_batch])[:, -1]

    def response(self, texts, topk=5):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], 1, topk)  # -> AutoRegressiveDecoder/random_sample @ ren ma 2022.11.11
        return tokenizer.decode(results[0])

    def response_batch_raw(self, texts, n_perbatch,topk=5):
        # make sure the input text in same length， texts is the inputs*batch_size，texts should be shuffled, in order to gen different result
        # 因为只产生评论，所以segment_id全为0
        token_ids_batch = []
        segment_ids_batch = []
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            segs = [0] * len(ids)
            token_ids_batch.append([tokenizer._token_start_id] + ids)
            segment_ids_batch.append([0] + segs)
        # 需要padding
        token_ids_batch = sequence_padding(token_ids_batch)
        segment_ids_batch = sequence_padding(segment_ids_batch)
        results = self.random_sample_batch([np.asarray(token_ids_batch), np.asarray(segment_ids_batch)], n_perbatch, topk, tokenizer=tokenizer)
        res = []
        for item in results:
            if item is not None:
                res.append(tokenizer.decode(item))
            else:
                res.append('')
        return res

    def response_batch(self, texts, batch_size, topk=5):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], batch_size, topk)
        res_final = []
        for res in results:
            res_final.append(tokenizer.decode(res))
        return res_final


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)
data_dir = '/data/renma/persona_chat/datasets/ready4training/KdConv'
def predict(df):
    """根据输入预测回复"""
    predict_nezha = []
    for x in tqdm(df['inputs4nezha']):
        predict_nezha.append(chatbot.response([x[0]]))
    df['predict_nezha'] = predict_nezha
    return df

test4nezha = pd.read_csv(data_dir + '/test4nezha.csv')
test4nezha['inputs4nezha'] = test4nezha['inputs4nezha'].apply(ast.literal_eval)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    test4nezha500 = predict(test4nezha[:500])
test4nezha500.to_excel(data_dir + '/test4nezha_withPredictions_1121.xlsx', index=False)

# print(chatbot.response([u'我最近天天吃火锅', u'我想到成都去玩']))
# b0 = time.time()
# # text = '美国暴力执法，黑人遭枪击，身中20枪，身亡。'
# text_list = [('美国大选','美国大选，乔·拜登正式成为2020年美国民主党总统候选人。卡玛拉·哈里斯成为2020年美国民主党副总统候选人，特朗普败选。'),
#              ('刘强东性侵','2018年9月2日，网传京东CEO刘强东在美国明尼苏达州因涉嫌性侵女大学生被捕。'),
#              ('华为被美封杀','2019年5月15日，美国总统特朗普签署行政令，禁止美国公司使用有对国家安全构成风险的公司制造的通讯设备。这项行政令宣布国家紧急状态，这是正式禁止与华为做生意的第一步。同一天，美国商务部宣布把华为及其子公司列入出口管制的“实体名单”。'),
#              ('青岛大虾','2015年10月4日，肖先生在青岛市乐陵路92号的“善德活海鲜烧烤家常菜”吃饭时遇到宰客事件引发网友热议。在吃饭前，曾详细询问过菜价，向老板确认过大虾38元究竟是一份还是一只，肖先生称当时老板说的是38元一份。但吃完饭后，老板却称大虾价格为38元一只。'),
#              ('马云外滩演讲','2020年10月24日，蚂蚁集团即将上市前夕，马云在上海第二届外滩金融峰会上的一番犀利讲话，引发中国金融圈震动。马云约20分钟的演讲，抛出很多带有一些哲理意味的格言警句式的论断，代表性说法包括：“今天银行延续的还是当铺思想”，“好的创新不怕监管，但是怕昨天的方式去监管”。11月2日，马云突然遭到中国四大金融监管机构约谈。次日，上海证券交易所发布关于暂缓蚂蚁集团科创板上市的决定。')]

# for fname, text in text_list:
#     res_set = defaultdict()
#     text = ['美国大选，乔·拜登正式成为2020年美国民主党总统候选人。卡玛拉·哈里斯成为2020年美国民主党副总统候选人，特朗普败选。', '京东CEO刘强东在美国明尼苏达州因涉嫌性侵女大学生被捕。', '特朗普政府禁止美国公司使用华为公司制造的通讯设备。','上海证券交易所发布关于暂缓蚂蚁集团科创板上市的决定。']
#     text_dict = {i:[] for i in range(len(text))}
#     n_perbatch = 2
#     while len(res_set)<100000:
#         response = chatbot.response_batch_raw(text,n_perbatch=n_perbatch) # u'你这样会失去我的', u'失去了又能怎样'
#         # 输入与结果对应
#         for i in text_dict:
#             text_dict[i].extend(response[i*n_perbatch:(i+1)*n_perbatch])
#             print('text_dict', set(text_dict[i]))

#         if not isinstance(response,list):
#             response = [response]
#         for resp in response:
#             if resp not in res_set:
#                 res_set[resp] = 1
#             else:
#                 res_set[resp] += 1
#         print('len res_set {},{:.2f}'.format(len(res_set),time.time()-b0))
#         if len(res_set)%100 == 0:
#             print('{} items cost {:.2f}s'.format(len(res_set),time.time()-b0))
#         """
#         回复是随机的，例如：那你还爱我吗 | 不知道 | 爱情是不是不能因为一点小事就否定了 | 我会一直爱你，你一个人会很辛苦 | 等等。
#         """

#     with open(os.path.join(file_root,fname+'_samples.txt'),'w') as fw:
#         fw.write('topic: {}'.format(text) + '\n')
#         res_set = dict(sorted(res_set.items(),key=lambda x:x[1], reverse=True))
#         for k,v in res_set.items():
#             fw.write(k + ' ' + str(v) + '\n')