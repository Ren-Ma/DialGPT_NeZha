#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2020/11/4 9:53 上午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import json
import pandas as pd

file_root = os.path.dirname(__file__)

def load_json(json_file):
    with open(json_file,'r') as f:
        data_dict = json.load(f)
    return data_dict

def load_csv(csv_file):
    df = pd.read_csv(csv_file,engine='python')
    df.fillna('-1')
    return df

def comment_reverse(geners, df):
    for gener in geners:
        gener_pd = df['GENRES'].apply(genre_filter,**{'genre':gener})
        df[gener] = gener_pd
    df.to_csv(os.path.join(file_root, '../../data/comments_reverse.csv'))

def genre_filter(df,genre):
    if df:
        if genre in str(df):
            return genre
        else:
            return ''
    else:
        return 'NAN'

def run(csv_file, person_tag_csv, json_file):
    df = load_csv(csv_file)
    df = df.dropna(subset=["CONTENT"])
    df = df.dropna(subset=['COMMENT_ID'])
    tag = load_csv(person_tag_csv)
    df['COMMENT_ID'] = df['COMMENT_ID'].apply(int).apply(str)
    tag['COMMENT_ID'] = tag['COMMENT_ID'].apply(int).apply(str)
    df = pd.merge(df, tag, how='outer', on='COMMENT_ID')
    # drop一些没法返回的值
    df = df.dropna(subset=['COMMENT_ID'])
    df = df.dropna(subset=['COMMENT_TIME'])
    df = df.dropna(subset=['RATING'])
    df['RATING'] = df['RATING'].apply(int).apply(str)
    geners = load_json(json_file)
    comment_reverse(geners,df)

if __name__=="__main__":
    json_file = os.path.join(file_root, '../../data/genres_select.json')
    csv_file = os.path.join(file_root, '../../data/merge.csv')
    person_tag = os.path.join(file_root, '../../data/personality_tag.csv')
    data = pd.read_csv(person_tag)
    run(csv_file, person_tag, json_file)