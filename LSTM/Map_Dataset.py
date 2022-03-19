from torch.utils.data import Dataset

"""construct the mapper used by torch.utils.data.DataLoader"""
class DatasetMaper(Dataset):
	
	"""initialization"""
	def __init__(self, x, y):
		self.x = x
		self.y = y
	
	"""return length"""
	def __len__(self):
		return len(self.x)
		
	"""return one element"""
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]