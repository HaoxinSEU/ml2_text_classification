import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import re

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

"""Import the data and then do preprocess"""
class Text_Processing:
	"""Initialization"""
	def __init__(self, maxLength, maxWords):
		self.maxLength = maxLength
		self.maxWords = maxWords
	
	"""load the train data, and split into train and evaluation set"""
	def load_train_data(self):
		data_train_neg = pd.read_csv('../datasets/train_neg_full.txt', sep = '\r', names = ['text'], header=None, encoding = 'utf-8')
		data_train_neg.insert(data_train_neg.shape[1], "labels", 0)

		data_train_pos = pd.read_csv('../datasets/train_pos_full.txt', sep = '\r', names = ['text'], header=None, encoding = 'utf-8')
		data_train_pos.insert(data_train_pos.shape[1], "labels", 1)

		data_train = pd.concat([data_train_neg,data_train_pos], ignore_index=True)
		data_train = data_train.sample(frac=1).reset_index(drop=True)

		data_eval = data_train.iloc[2400000:-1,:]
		data_eval = data_eval.reset_index(drop=True)
		data_train = data_train.iloc[0:2400000,:]

		self.train_X = data_train['text'].values
		self.train_Y = data_train['labels'].values

		self.eval_X = data_eval['text'].values
		self.eval_Y = data_eval['labels'].values

	"""tokenize"""
	def tokenize_words(self):
		self.tokens = Tokenizer(num_words=self.maxWords)
		self.tokens.fit_on_texts(self.train_X)

	def seq2token(self, X):
		seq = self.tokens.texts_to_sequences(X)
		return sequence.pad_sequences(seq, maxlen = self.maxLength)

	"""load the test data"""
	def load_test_data(self):
		data_test = pd.read_csv('../datasets/test_data.txt', sep = '\r', names = ['text'], header=None, encoding = 'utf-8')

		self.test_X = data_test['text'].values
		

