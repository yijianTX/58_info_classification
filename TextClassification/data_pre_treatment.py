#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json, os, re
import jieba
import jieba.posseg as pg

# 用于统计每个类目的数据个数
cate_data_count = {}

cate_id_map = {
	u"物流/仓储" : 0,
	u"保险" : 1,
	u"美容/美发" : 2,
	u"影视/娱乐/休闲" : 3,
	u"法律" : 4,
	u"物业管理" : 5,
	u"质控/安防" : 6,
	u"淘宝职位" : 7,
	u"客服" : 8,
	u"美术/设计/创意" : 9,
	u"化工" : 10,
	u"酒店" : 11,
	u"教育培训" : 12,
	u"房产中介" : 13,
	u"高级管理" : 14,
	u"环保/能源" : 15,
	u"编辑/出版/印刷" : 16,
	u"服装/纺织/食品" : 17,
	u"家政保洁/安保" : 18,
	u"贸易/采购" : 19,
	u"运动健身" : 20,
	u"电子/电气" : 21,
	u"旅游" : 22,
	u"广告/会展/咨询" : 23,
	u"制药/生物工程" : 24,
	u"普工/技工" : 25,
	u"机械/仪器仪表" : 26,
	u"建筑" : 27,
	u"餐饮" : 28,
	u"超市/百货/零售" : 29,
	u"保健按摩" : 30,
	u"金融/银行/证券/投资" : 31,
	u"市场/媒介/公关" : 32,
	u"销售" : 33,
	u"司机/交通服务" : 34,
	u"计算机/互联网/通信" : 35,
	u"农/林/牧/渔业" : 36,
	u"财务/审计/统计" : 37,
	u"生产管理/研发" : 38,
	u"汽车制造/服务" : 39,
	u"翻译" : 40,
	u"人事/行政/后勤" : 41,
	u"医院/医疗/护理" : 42
}

# jieba词典中需要的词性
pos_jieba_needed = ['n', 'ns', 'nt', 'nz', 'v', 'vn', 'a', 'an', 'j', 'l', 't', 'm']

# 人工标签词典中需要的词性
pos_manual_needed = ["flbt", "flbx", "jy", "gt", "fljq", "flly", "zw", "cs", "hy", "jn", "pp", "nl", "xh", "rj", "sg", "wg", "xb", "xg", "xx", "yxz", "xl", "zs", "zy"]

# 数据中的html标签
html_pat = ur"</?(?:head|body|pstyle|p|strong|spanstyle|span|brstyle|br|b|palign|fontcolor|font|table|tr)[/\w\s:;!@#$%\"\'\?\.,\^\&\*\(\)\-\+=微软雅黑]*/?>"

class Pretreatment(object):

	def __init__(self, data_dir_path, dict_dir_path, sentence_max_length, use_segment = False):
		self.data_dir_path = data_dir_path
		self.stopwords_set = self.load_stopwords(os.path.join(dict_dir_path, "stopwords.txt"))
		self.sentence_max_length = sentence_max_length
		self.use_segment = use_segment
		if self.use_segment:
			self.pos_set = set(pos_jieba_needed + pos_manual_needed)
			jieba.load_userdict(dict_dir_path, "manual_dict.txt")


	def load_stopwords(self, file_path):
		res_set = set()
		with open(file_path) as file:
			for line in file:
				line = line.strip().lower()
				res_set.add(line.decode("utf-8"))
		return res_set

	def load_data(self):
		"""
		用于生成数据，use_segment参数用于控制是否需要分词，即用字向量还是词向量建立模型。
		"""
		data_files = os.listdir(self.data_dir_path)
		data_files = [os.path.join(self.data_dir_path, item) for item in data_files]
		self.x = []
		self.y = []
		for file in data_files:
			with open(file) as f:
				# 先拿第一行检验cate id合法性及做相关设置
				test_line = re.sub(html_pat, "", f.readline().strip().lower())
				d_first = json.loads(test_line)
				if d_first["catename"] not in cate_id_map:
					break
				cate_data_count.setdefault(d_first["catename"], 1)
				cate_id = cate_id_map[d_first["catename"]]
				label = [0] * len(cate_id_map.keys())
				label[cate_id] = 1
				self.y.append(label)

				if self.use_segment:
					_ = pg.cut(d_first["title"] + " " + d_first["content"])
					self.x.append([item.word.strip() for item in _ if len(item.word.strip()) and item.flag in self.pos_set and item.word not in self.stopwords_set])
				else:
					_ = d_first["title"] + d_first["content"]
					self.x.append([item.strip() for item in list(_) if len(item.strip()) and item not in self.stopwords_set])

				# 继续读完文件
				for line in f:
					line = re.sub(html_pat, "", line.strip().lower())
					try:
						d_temp = json.loads(line)
					except:
						continue
					cate_data_count[d_temp["catename"]] += 1
					self.y.append(label)

					if self.use_segment:
						temp = pg.cut(d_temp["title"] + " " + d_temp["content"])
						self.x.append([item.word.strip() for item in temp if len(item.word.strip()) and item.flag in self.pos_set and item.word not in self.stopwords_set])
					else:
						temp = d_temp["title"] + d_temp["content"]
						self.x.append([item.strip() for item in list(temp) if len(item.strip()) and item not in self.stopwords_set])

	def map_vocabs(self):
		# 用于生成词语和id之间的映射。
		self.d_vocab = {}
		idx = 0
		self.d_vocab["<pad>"] = idx
		for sentence in self.x:
			for word in sentence:
				if word not in self.d_vocab:
					idx += 1
					self.d_vocab[word] = idx

	def generate_input(self):
		# 转化数据为vocab index的向量。
		temp = [[self.d_vocab[word] for word in sent if word in self.d_vocab] for sent in self.x]
		self.x = []
		for sent in temp:
			sent.extend([ self.d_vocab["<pad>"] ] * (self.sentence_max_length - len(sent)))
			self.x.append(sent)

if __name__ == "__main__":
	pretreatment = Pretreatment(
		data_dir_path = "C:/Users/jinhuangyu/Desktop/58_info_classification/TextClassification/data", 
		dict_dir_path = "C:/Users/jinhuangyu/Desktop/58_info_classification/TextClassification/dict", 
		sentence_max_length = 100,
		use_segment = False
		)
	pretreatment.load_data()
	print cate_data_count
	pretreatment.map_vocabs()
	pretreatment.generate_input()
	print pretreatment.x[100]


	
