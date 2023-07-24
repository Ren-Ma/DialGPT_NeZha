#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2020/11/16 11:37 上午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import random
import sys
import time

sys.path.append('.')
import numpy as np

from src.TextGen.bert_layers.snippets import AutoRegressiveDecoder, sequence_padding
import os
from collections import defaultdict
import pandas as pd
import tensorflow as tf

file_root = os.path.dirname(__file__)

class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    def __init__(self, model, tokenizer, start_id, end_id, maxlen, minlen=None):
        super(ChatBot,self).__init__(**{"start_id":start_id,"end_id":end_id,"maxlen":maxlen,"minlen":minlen})
        self.model = model
        self.tokenizer = tokenizer

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        preds = self.model.predict([token_ids, segment_ids])[:, -1]
        return preds

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict_batch(self, inputs_batch, output_ids, states):
        token_ids_batch, segment_ids_batch = inputs_batch
        token_ids_batch = np.concatenate([token_ids_batch, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids_batch[0, -1]
        segment_ids_batch = np.concatenate([segment_ids_batch, curr_segment_ids], 1)
        preds = self.model.predict_on_batch([token_ids_batch, segment_ids_batch])[:, -1]
        return preds

    def response(self, texts, topk=5):
        token_ids, segment_ids = [self.tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = self.tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], 1, topk)
        return self.tokenizer.decode(results[0])

    def response_batch_raw(self, texts, n_perbatch, topk=5, tokenizer=None):
        # make sure the input text in same length， texts is the inputs*batch_size，texts should be shuffled, in order to gen different result
        # 因为只产生评论，所以segment_id全为0
        token_ids_batch = []
        segment_ids_batch = []
        for i,text in enumerate(texts):
            ids = self.tokenizer.encode(text)[0][1:]
            segs = [0] * len(ids)
            token_ids_batch.append([self.tokenizer._token_start_id]+ids)
            segment_ids_batch.append([0]+segs)
        # 需要padding
        token_ids_batch = sequence_padding(token_ids_batch)
        segment_ids_batch = sequence_padding(segment_ids_batch)
        results = self.random_sample_batch([np.asarray(token_ids_batch), np.asarray(segment_ids_batch)], n_perbatch, topk, tokenizer=tokenizer)
        res = []
        for item in results:
            if item is None:
                continue
            res.append(self.tokenizer.decode(item) if item.any() else '')
        return res

    def response_batch(self, texts, batch_size, topk=5):
        token_ids, segment_ids = [self.tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = self.tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], batch_size, topk)
        res_final = []
        for res in results:
            res_final.append(self.tokenizer.decode(res))
        return res_final
