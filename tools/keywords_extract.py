#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2020/12/8 12:08 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

from __future__ import absolute_import
import os
import random
import requests
import re
import time
import json
import jieba
import jieba.posseg
from LAC import LAC

# 装载LAC模型
# PER	人名	LOC	地名	ORG	机构名	TIME	时间
# n	普通名词 nz	其他专名  s	处所名词	nw	作品名
entity_type = {'PER':'人物','LOC':'地点','ORG':'机构','TIME':'时间','nz':'专有名词','s':'专有名词','nw':'专有名词'}
lac = LAC(mode='lac')

summ_url = "113.31.111.86:48068"

file_root = os.path.dirname(__file__)
_get_abs_path = jieba._get_abs_path

DEFAULT_IDF = os.path.join(file_root,"idf.txt")
STOP_WORD_PATH = os.path.join(file_root,'停用词.txt')

class KeywordExtractor(object):
    def __init__(self,stop_words_path):
        self.stop_words = set((
            "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are",
            "by", "be", "as", "on", "with", "can", "if", "from", "which", "you", "it",
            "this", "then", "at", "have", "all", "not", "one", "has", "or", "that"
        ))
        self.set_stop_words(stop_words_path=stop_words_path)

    def set_stop_words(self, stop_words_path):
        abs_path = _get_abs_path(stop_words_path)
        if not os.path.isfile(abs_path):
            raise Exception("jieba: file does not exist: " + abs_path)
        content = open(abs_path, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.add(line)

    def extract_tags(self, *args, **kwargs):
        raise NotImplementedError


class IDFLoader(object):

    def __init__(self, idf_path=None):
        self.path = ""
        self.idf_freq = {}
        self.median_idf = 0.0
        if idf_path:
            self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path):
        if self.path != new_idf_path:
            self.path = new_idf_path
            content = open(new_idf_path, 'rb').read().decode('utf-8')
            self.idf_freq = {}
            for line in content.splitlines():
                word, freq = line.strip().split(' ')
                self.idf_freq[word] = float(freq)
            self.median_idf = sorted(
                self.idf_freq.values())[len(self.idf_freq) // 2]

    def get_idf(self):
        return self.idf_freq, self.median_idf


class TFIDF(KeywordExtractor):

    def __init__(self, idf_path=None):
        super(TFIDF,self).__init__(stop_words_path=STOP_WORD_PATH)
        self.tokenizer = jieba.dt
        self.postokenizer = jieba.posseg.dt
        # self.stop_words = self.STOP_WORDS.copy() #已经在KeywordExtractor定义
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def set_idf_path(self, idf_path):
        new_abs_path = _get_abs_path(idf_path)
        if not os.path.isfile(new_abs_path):
            raise Exception("jieba: file does not exist: " + new_abs_path)
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def extract_tags(self, sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False):
        """
        Extract keywords from sentence using TF-IDF algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v','nr'].
                        if the POS of w is not in this list,it will be filtered.
            - withFlag: only work with allowPOS is not empty.
                        if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        if allowPOS:
            allowPOS = frozenset(allowPOS)
            words = self.postokenizer.cut(sentence)
        else:
            words = self.tokenizer.cut(sentence)
        freq = {}
        for w in words:
            if allowPOS:
                if w.flag not in allowPOS:
                    continue
                elif not withFlag:
                    w = w.word
            wc = w.word if allowPOS and withFlag else w
            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                continue
            freq[w] = freq.get(w, 0.0) + 1.0
        total = sum(freq.values())
        for k in freq:
            kw = k.word if allowPOS and withFlag else k
            freq[k] *= self.idf_freq.get(kw, self.median_idf) / total

        if withWeight:
            # 修改返回topk个关键字，但是不改变关键字在文中的位置
            # tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
            tags = list(freq)
        else:
            # tags = sorted(freq, key=freq.__getitem__, reverse=True)
            tags = list(freq)
        if topK:
            return tags[:topK]
        else:
            return tags

tfidf = TFIDF()

def parse_paragraph(text):
    print('parsing paragraph ...')
    sent_list = re.split(r'\n\n\n|\r\n\r\n|\n\n|\n',text)
    sent_list_final = []
    for i in sent_list:
        tmp = i.replace(' ','').strip(',，。."?、？：！”\n')
        if len(tmp) > 10:
            sent_list_final.append(tmp)
    print('finish parsing paragraph!')
    return sent_list_final

def parse_sent(sent_list):
    b0 = time.time()
    topic_final = []
    keyword_final = []
    print('parsing sentence...')
    for sent in sent_list:
        summ_content = ''
        try:
            summ = requests.post('http://{}/abstract'.format(summ_url), json={'text': sent}, timeout=3)
            data = json.loads(summ.content)['data']
            summ_content = data['abstract'].replace('[UNK] ', '').replace(' ', '').strip(',，。."?、？：！”')
        except requests.exceptions.RequestException as e:
            print(e)
        res = tfidf.extract_tags(sent, allowPOS=['n', 'v'])
        keyword_final.extend(res)
        entity = lac.run(sent)
        entity_final = set()
        for ent, tag in zip(*entity):
            ent = ent.replace(' ','').strip(',，。."?、？：！”\n')
            if tag in entity_type and len(ent) >= 2:
                entity_final.add((ent, entity_type[tag]))
        entity = [i[0] for i in entity_final]
        key_window = []
        for i in range(len(res[::3])):
            tmp = res[i * 3:min(i * 3 + 6, len(res))]
            if len(tmp) == 6:
                key_window.append(tmp)
                if entity:
                    e = random.choice(entity)
                    t = e + ',' + summ_content + ',' + ''.join(tmp)
                    topic_final.append(t)
                else:
                    topic_final.append(summ_content + ',' + ''.join(tmp))
            elif i == 0 and len(''.join(tmp)) < 13:
                if entity:
                    e = random.choice(entity)
                    topic_final.append(e + ',' + summ_content + ',' + ''.join(tmp))
                else:
                    topic_final.append(summ_content + ',' + ''.join(tmp))

    print(topic_final)
    print('parse sentences cost {:.2f}'.format(time.time() - b0))
    return topic_final, keyword_final

def run_keywords(text):
    if len(text) < 50:
        keywords_list = tfidf.extract_tags(text, allowPOS=['n', 'v'])
        topic_list = [text]
        if len(keywords_list) >= 2:
            topic_list = [text+','+''.join(random.sample(keywords_list,2))]
    else:
        sent_list = parse_paragraph(text)
        topic_list, keywords_list = parse_sent(sent_list)
    return topic_list, keywords_list

if __name__=='__main__':
    text = """国家主席习近平发表二〇二一年新年贺词
        \n\n2020年是极不平凡的一年。面对突如其来的新冠肺炎疫情，我们以人民至上、生命至上诠释了人间大爱，用众志成城、坚忍不拔书写了抗疫史诗。在共克时艰的日子里，有逆行出征的豪迈，有顽强不屈的坚守，有患难与共的担当，有英勇无畏的牺牲，有守望相助的感动。从白衣天使到人民子弟兵，从科研人员到社区工作者，从志愿者到工程建设者，从古稀老人到“90后”、“00后”青年一代，无数人以生命赴使命、用挚爱护苍生，将涓滴之力汇聚成磅礴伟力，构筑起守护生命的铜墙铁壁。一个个义无反顾的身影，一次次心手相连的接力，一幕幕感人至深的场景，生动展示了伟大抗疫精神。平凡铸就伟大，英雄来自人民。每个人都了不起！向所有不幸感染的病患者表示慰问！向所有平凡的英雄致敬！我为伟大的祖国和人民而骄傲，为自强不息的民族精神而自豪！
        \n\n艰难方显勇毅，磨砺始得玉成。我们克服疫情影响，统筹疫情防控和经济社会发展取得重大成果。“十三五”圆满收官，“十四五”全面擘画。新发展格局加快构建，高质量发展深入实施。我国在世界主要经济体中率先实现正增长，预计2020年国内生产总值迈上百万亿元新台阶。粮食生产喜获“十七连丰”。“天问一号”、“嫦娥五号”、“奋斗者”号等科学探测实现重大突破。海南自由贸易港建设蓬勃展开。我们还抵御了严重洪涝灾害，广大军民不畏艰险，同心协力抗洪救灾，努力把损失降到了最低。我到13个省区市考察时欣喜看到，大家认真细致落实防疫措施，争分夺秒复工复产，全力以赴创新创造，神州大地自信自强、充满韧劲，一派只争朝夕、生机勃勃的景象。
        \n\n2020年，全面建成小康社会取得伟大历史性成就，决战脱贫攻坚取得决定性胜利。我们向深度贫困堡垒发起总攻，啃下了最难啃的“硬骨头”。历经8年，现行标准下近1亿农村贫困人口全部脱贫，832个贫困县全部摘帽。这些年，我去了全国14个集中连片特困地区，乡亲们愚公移山的干劲，广大扶贫干部倾情投入的奉献，时常浮现在脑海。我们还要咬定青山不放松，脚踏实地加油干，努力绘就乡村振兴的壮美画卷，朝着共同富裕的目标稳步前行。
        \n\n今年，我们隆重庆祝深圳等经济特区建立40周年、上海浦东开发开放30周年。置身春潮涌动的南海之滨、绚丽多姿的黄浦江畔，令人百感交集，先行先试变成了示范引领，探索创新成为了创新引领。改革开放创造了发展奇迹，今后还要以更大气魄深化改革、扩大开放，续写更多“春天的故事”。
        \n\n大道不孤，天下一家。经历了一年来的风雨，我们比任何时候都更加深切体会到人类命运共同体的意义。我同国际上新老朋友进行了多次通话，出席了多场“云会议”，谈得最多的就是和衷共济、团结抗疫。疫情防控任重道远。世界各国人民要携起手来，风雨同舟，早日驱散疫情的阴霾，努力建设更加美好的地球家园。
        \n\n2021年是中国共产党百年华诞。百年征程波澜壮阔，百年初心历久弥坚。从上海石库门到嘉兴南湖，一艘小小红船承载着人民的重托、民族的希望，越过急流险滩，穿过惊涛骇浪，成为领航中国行稳致远的巍巍巨轮。胸怀千秋伟业，恰是百年风华。我们秉持以人民为中心，永葆初心、牢记使命，乘风破浪、扬帆远航，一定能实现中华民族伟大复兴。
        \n\n站在“两个一百年”的历史交汇点，全面建设社会主义现代化国家新征程即将开启。征途漫漫，惟有奋斗。我们通过奋斗，披荆斩棘，走过了万水千山。我们还要继续奋斗，勇往直前，创造更加灿烂的辉煌！
        \n\n此时此刻，华灯初上，万家团圆。新年将至，惟愿山河锦绣、国泰民安！惟愿和顺致祥、幸福美满！"""
    t = run_keywords(text)
