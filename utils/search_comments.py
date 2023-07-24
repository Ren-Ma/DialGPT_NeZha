#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2020/11/4 6:32 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import pandas as pd
import time
import random
import json

file_root = os.path.dirname(__file__)

def search_comments(df, genre_list, sentiment_list, person_tag_list, topn=30):
    df_select = df.loc[df['GENRES'].isin(genre_list) & df['SENTIMENT'].isin(sentiment_list) & df['内外向'].isin(person_tag_list) & df['神经质'].isin(person_tag_list)]
    if len(df_select)>=topn:
        print('first res total length {}'.format(len(df_select)))
        return df_select.sample(n=topn)
    else:
        df_select = df.loc[df['GENRES'].isin(genre_list) & df['SENTIMENT'].isin(sentiment_list)]
        print('second res total length {}'.format(len(df_select)))
        return df_select.sample(n=topn) if len(df_select)>=topn else df_select

if __name__=='__main__':
    person_df = pd.read_csv(os.path.join(file_root, '../../data/users.csv'))
    reverse_path = os.path.join(file_root, '../../data/comments_reverse.csv')
    data = pd.read_csv(reverse_path)
    genre = ['恐怖', '剧情']
    sentiment = ['正向','负向'] # 情绪
    person_tag = ['内向', '外向','神经质高','情绪稳定']
    b0 = time.time()
    df_select = search_comments(data, genre, sentiment, person_tag, topn=30)
    submit_time_list = sorted(df_select['COMMENT_TIME'].tolist(), reverse=False)
    result = {'comment': [], 'submit_time': [], 'rating': [], 'person_id': [], 'person_nickname':[]}
    result['person_id'] = random.sample(range(0, 1000), len(df_select))
    result['person_nickname'] = person_df['USER_NICKNAME'].sample(len(df_select)).to_list()
    for i, (idx, row) in enumerate(df_select.iterrows()):
        result['comment'].append(row['CONTENT'])
        result['submit_time'].append(submit_time_list[i])
        result['rating'].append(row['RATING'])
    print('total costs {:.2f}s'.format(time.time() - b0))
    print(json.dumps(result,indent=2,ensure_ascii=False))