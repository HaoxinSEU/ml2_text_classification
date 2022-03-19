import numpy as np
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from RNN_LSTM import LSTM_Classification
from Text_Processing import Text_Processing
from Map_Dataset import DatasetMaper

import time

"""Use to start the training and inference"""
class Run:
	"""initialization"""
	def __init__(self, batchSize, hiddenDim, LSTMLayers, maxWords, maxLength):
		
		# do initialization for data
		self.data_init(maxLength, maxWords)
		# the batch size of each iter
		self.batchSize = batchSize
		# create the LSTM RNN
		self.model = LSTM_Classification(batchSize, hiddenDim, LSTMLayers, maxWords)
		print("Create a LSTM model")


	"""Use Text_Processing to load data"""
	def data_init(self, maxLength, maxWords):
		self.textProcessing = Text_Processing(maxLength, maxWords)
		# load the training data
		self.textProcessing.load_train_data()
		# load the test data
		self.textProcessing.load_test_data()

		
	"""tokenize training data"""
	def train_data_processing(self):
		# tokenize
		self.textProcessing.tokenize_words()

		self.train_X = self.textProcessing.seq2token(self.textProcessing.train_X)
		self.train_Y = self.textProcessing.train_Y
		
	
	"""tokenize evaluation data"""
	def eval_data_processing(self):
		self.eval_X = self.textProcessing.seq2token(self.textProcessing.eval_X)
		self.eval_Y = self.textProcessing.eval_Y


	"""tokenize test data"""
	def test_data_processing(self):
		self.test_X = self.textProcessing.seq2token(self.textProcessing.test_X)

	
	def train(self, learningRate, epochNum):
		# start the timer
		starttime = time.time()

		# pre-process the data
		self.train_data_processing()

		# use torch.utils.data.DataLoader
		train_Set = DatasetMaper(self.train_X, self.train_Y)
		self.train_Loader = DataLoader(train_Set, batch_size=self.batchSize)
		

		optimizer = optim.RMSprop(self.model.parameters(), lr=learningRate)

		# run the SGD in each epoch
		for epoch in range(0, epochNum):

			self.model.train()

			i = 0

			# mini-batch to update the network
			for batch_train_X, batch_train_Y in self.train_Loader:
				
				x = batch_train_X.type(torch.LongTensor)
				y = batch_train_Y.type(torch.FloatTensor).unsqueeze(1)
				
				y_pred = self.model(x)
				
				# loss function
				loss = F.binary_cross_entropy(y_pred, y)
				
				optimizer.zero_grad()
				
				loss.backward()
				
				optimizer.step()

				# print information
				i = i + 1
				if i % 10000 == 0:
					endtime_batch = time.time()
					print("finish batch: ", i)
					print("training already use time: ", endtime_batch - starttime)
		
		endtime_train = time.time()
		print("training consume time: ", endtime_train - starttime)
		
		# begin evaluation
		self.eval_data_processing()
		eval_Set = DatasetMaper(self.eval_X, self.eval_Y)
		self.eval_Loader = DataLoader(eval_Set)
		pred_eval_Y = self.evaluation()
		eval_accur = self.cal_accuracy(self.eval_Y, pred_eval_Y)

		endtime_eval = time.time() 
		print("evaluation consume time: ", endtime_eval - endtime_train)
		print("The accuracy on evaluation set is: ", eval_accur)

	"""Use the network to do prediction"""
	def evaluation(self):
		prediction = []
		self.model.eval()
		with torch.no_grad():
			#for batch_eval_X, batch_eval_Y in self.eval_Loader:
			for batch_eval_X, _ in self.eval_Loader:
				x = batch_eval_X.type(torch.LongTensor)
				
				y_pred = self.model(x)
				prediction += list(y_pred.detach().numpy())
				
		return prediction


	"""Calculate the accuracy of prediction"""
	def cal_accuracy(self, eval_y, pred_eval_y):
		true_pos = 0
		true_neg = 0
		
		# counter tn, tp to calculate F1 score
		for true, pred in zip(eval_y, pred_eval_y):
			if (pred > 0.5) and (true == 1):
				true_pos += 1
			elif (pred < 0.5) and (true == 0):
				true_neg += 1
			else:
				pass
				
		print("The number of true positive is: ", true_pos)
		print("The number of true negative is: ", true_neg)

		return (true_pos + true_neg) / len(eval_y)


	"""Use the trained network to do inference, i.e. classification"""
	def inference(self):
		prediction = []
		starttime_test = time.time()

		self.test_data_processing()
		self.model.eval()
		self.test_Loader = DataLoader(self.test_X)

		# use the model to do prediction
		with torch.no_grad():

			for sample_test_X in self.test_Loader:
				x = sample_test_X.type(torch.LongTensor)

				y_pred = self.model(x)
				prediction += list(y_pred.detach().numpy())

		# convert results from [0,1] to {-1,1}
		predict_res = np.array(prediction)
		predict_res[np.where(predict_res >= 0.5)] = 1
		predict_res[np.where(predict_res < 0.5)] = -1


		endtime_test = time.time()
		print("prediction costs time: ", endtime_test - starttime_test)

		# write the output file
		test_id = np.arange(1, 10001)

		print("writing results to csv...")
		with open('../prediction.csv', 'w') as csvfile:
			fieldnames = ['Id', 'Prediction']
			writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
			writer.writeheader()
			for r1, r2 in zip(test_id, predict_res):
				writer.writerow({'Id':int(r1),'Prediction':int(r2)})
		print("finished!")



if __name__ == "__main__":
	run = Run(32, 32, 2, 1000, 32)
	run.train(1e-3, 1)
	run.inference()
