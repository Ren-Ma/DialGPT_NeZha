#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2020/11/3 3:38 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os

import pandas as pd
import numpy as np
import re
import json
from tqdm import tqdm
from collections import Counter
import multiprocessing

file_root = os.path.dirname(__file__)

def read_csv(data_path,tag='movie'):
    if tag=='comment':
        df = pd.read_csv(data_path,engine='python', sep="\x01")
    else:
        df = pd.read_csv(data_path, engine='python')

    df = df.fillna('')
    if 'GENRES' in df:
        df['GENRES'] = df['GENRES'].replace(',','').replace('\n','').str.split('/',expand=False)
    if 'TAGS' in df:
        df['TAGS'] = df['TAGS'].replace(',','').replace('\n','').str.split('/',expand=False)
    if 'CONTENT' in df:
        df['CONTENT'] = df['CONTENT'].replace(',','').replace('\n','')
    if 'RATING' in df:
        # 评论为空的全部过滤
        df['RATING'] = df['RATING'].replace(to_replace='',value=np.nan,regex=True,inplace=False)
        df = df[df['RATING'].notnull()]
    return df

def find_id(line):
    match_id = r'^"[0-9]{3,12}".*?$'
    id = re.match(match_id, line)
    if id:
        return 1
    else:
        return 0

def combine_data(data,id_idx):
    data_final = []
    start = 0
    for i, line in tqdm(enumerate(data)):
        if i in id_idx and i > start:
            line_mod = ''.join(data[start:i]).replace('\n', '')
            data_final.append(line_mod)

    if start <= len(data) - 1:
        line_mod = ''.join(data[start:len(data)]).replace('\n', '')
        data_final.append(line_mod)

    return data_final

def read_txt_stream(data):
    match_id = r'^"[0-9]{3,12}".*?$'
    id_idx = []
    for i, line in enumerate(data):
        id = re.match(match_id,line)
        if id:
            id_idx.append(i)
    print('id_idx 长度',len(id_idx))

    data_final = []
    start = 0
    for i,line in tqdm(enumerate(data)):
        if i in id_idx and i > start:
            line_mod = ''.join(data[start:i]).replace('\n','')
            data_final.append(line_mod)

    if start <= len(data)-1:
        line_mod = ''.join(data[start:len(data)]).replace('\n','')
        data_final.append(line_mod)

    return data_final

def find_target(data_final, match_header):
    num = len(match_header)
    match_comma = r'.*?(".*?")' * num + '.*?$'
    unmatch = 0
    data_out = {item: [] for item in match_header}
    for line in tqdm(data_final):
        match = re.match(match_comma, line)
        try:
            assert len(match.groups()) == num
            for idx, header in enumerate(match_header):
                data_out[header].append(match.group(idx + 1).replace(',', '').replace('\n', ''))
        except Exception as e:
            unmatch += 1
            print('{} unmatched, line {}'.format(unmatch, line))

    data_out = pd.DataFrame(data_out)
    return data_out


def merge_movie_comment(df_movie, df_comment):
    movie_select_column = ["MOVIE_ID", "NAME", "ALIAS", "GENRES", "TAGS"]
    comments_select_column = ["COMMENT_ID", "MOVIE_ID", "CONTENT", "COMMENT_TIME", "RATING"]
    df_movie_select = pd.DataFrame(df_movie, columns=movie_select_column)
    df_comment_select = pd.DataFrame(df_comment,columns=comments_select_column)
    df_merge = pd.merge(df_comment_select, df_movie_select, how='outer', on='MOVIE_ID')
    # 内容为movie_id and comment_id
    category = Counter()
    tag = Counter()
    for i in df_merge['GENRES'].to_list():
        if isinstance(i,list):
            for j in i:
                if j:
                    category[j] += 1
        elif isinstance(i,str):
            for j in i.replace(',','').replace('\n','').split('/'):
                if j:
                    category[j] += 1

    for i in df_merge['TAGS'].to_list():
        if isinstance(i,list):
            for j in i:
                if j:
                    tag[j] += 1
        elif isinstance(i,str):
            for j in i.replace(',','').replace('\n','').split('/'):
                if j:
                    tag[j] += 1

    sentiment_pn = df_merge['RATING'].apply(lambda x: '正向' if isinstance(x,str) and x > '3' else '负向')
    df_merge['SENTIMENT'] = sentiment_pn
    with open(os.path.join(file_root,'../data/genres.json'), 'w') as ft:
        sorted_category = dict(sorted(category.items(), key=lambda item: item[1], reverse=True))
        categoryJson = json.dumps(sorted_category,indent=2,ensure_ascii=False)
        ft.write(categoryJson)
    with open(os.path.join(file_root,'../data/tags.json'), 'w') as fj:
        sorted_tag = dict(sorted(tag.items(), key=lambda item: item[1], reverse=True))
        tagJson = json.dumps(sorted_tag, indent=2, ensure_ascii=False)
        fj.write(tagJson)
    df_merge.to_csv(os.path.join(file_root,'../data/merge.csv'))


def read_txt_stream_fast(datapath,header):
    data_out = {item: [] for item in header}
    unmatch = 0
    with open(datapath,'r') as fr:
        data = fr.readlines()
        for i,line in tqdm(enumerate(data[1:])):
            if line.count('","') == len(header) - 1:
                line_list = line.split('","')
                for idx,item in enumerate(header):
                    try:
                        data_out[item].append(line_list[idx].replace('\n','').replace(',','').replace('"',''))
                    except Exception as e:
                        data_out[item].append('')
                        unmatch += 1
                        # print('line_list', line_list)
    print('unmatched',unmatch)
    return pd.DataFrame(data_out)


if __name__ == '__main__':
    process_num = 20
    data_path = os.path.join(file_root,'../data/movies.csv')
    data_path1 = os.path.join(file_root,'../data/comments.csv')
    data_path2 = os.path.join(file_root,'../data/personality_tag.csv')


    match_header1 = ["MOVIE_ID","NAME","ALIAS","ACTORS","COVER","DIRECTORS","DOUBAN_SCORE","DOUBAN_VOTES","GENRES","IMDB_ID","LANGUAGES","MINS","OFFICIAL_SITE","REGIONS","RELEASE_DATE","SLUG","STORYLINE","TAGS","YEAR","ACTOR_IDS","DIRECTOR_IDS"]
    match_header2 = ["COMMENT_ID","USER_MD5","MOVIE_ID","CONTENT","VOTES","COMMENT_TIME","RATING"]
    df_movie = read_txt_stream_fast(data_path,match_header1)
    df_comment = read_txt_stream_fast(data_path1,match_header2)
    merge_movie_comment(df_movie, df_comment)