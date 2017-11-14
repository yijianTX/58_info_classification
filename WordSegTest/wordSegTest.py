#!/usr/bin/env python
# -*- coding:utf-8 -*-

import jieba, json, os
import jieba.posseg as pg

cur_dir = os.path.dirname(__file__) or os.getcwd()

class Segment(object):
    
    def __init__(self, dict_path, stopwords_path, sensewords_path):
	jieba.load_userdict(dict_path)
	self.stopwords = self.load_words(stopwords_path)
	self.sensewords = self.load_words(sensewords_path)
	self.pos = set(["n", "nz", "vn", "v"])

    def load_words(self, file_path):
	s_words = set()
	with open(file_path) as file:
	    for line in file:
		line = line.strip()
		s_words.add(line)
	return s_words

    def word_cut(self, text):
	l_temp = []
	if not text or not len(text):
	    return l_temp
	for item in pg.cut(text):
	    if item.word not in self.stopwords and item.word not in self.sensewords and item.flag in self.pos:
		l_temp.append([item.word, item.flag])
	return json.dumps(l_temp, ensure_ascii = False)

if __name__ == "__main__":
    seg = Segment(cur_dir + "/user_dict.txt", cur_dir + "/stopwords.txt", cur_dir + "/sensewords.txt")
    print seg.word_cut("我想要测试一下这个东西到底有没有用,涉黄还是个名词么")
	
