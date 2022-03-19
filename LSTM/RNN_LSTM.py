import torch
import torch.nn as nn
import torch.nn.functional as F

# LSTM network
class LSTM_Classification(nn.Module):

	"""initialization"""
	def __init__(self, batchSize, hiddenDim, LSTMLayers, maxWords):
		super().__init__()
		
		self.batchSize = batchSize # batch size
		self.hiddenDim = hiddenDim # hidden dimension
		self.LSTMLayers = LSTMLayers # number of LSTM layers
		self.inputSize = maxWords # embedding dimention
		
		self.dropout = nn.Dropout()  # dropout
		self.embedding = nn.Embedding(self.inputSize, self.hiddenDim, padding_idx=0) # embedding
		self.lstm = nn.LSTM(input_size=self.hiddenDim, hidden_size=self.hiddenDim, num_layers=self.LSTMLayers, batch_first=True) # LSTM layers
		self.fc1 = nn.Linear(in_features=self.hiddenDim, out_features=257)
		self.fc2 = nn.Linear(257, 1)
		
	"""forward process"""
	def forward(self, x):
	
		h = torch.zeros((self.LSTMLayers, x.size(0), self.hiddenDim))
		c = torch.zeros((self.LSTMLayers, x.size(0), self.hiddenDim))
		
		torch.nn.init.xavier_normal_(h)
		torch.nn.init.xavier_normal_(c)

		out = self.embedding(x)
		out, (hidden, cell) = self.lstm(out, (h,c))
		out = self.dropout(out)
		out = torch.relu_(self.fc1(out[:,-1,:]))
		out = self.dropout(out)
		out = torch.sigmoid(self.fc2(out))

		return out