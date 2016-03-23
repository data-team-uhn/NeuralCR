import numpy as np

class DataReader:
	
	def __init__(self, data):
		self.data = data
		self.counter = 0

	def read_batch(self, triplets_name, labels_name, batch_size):
		triplets = self.data[triplets_name]
		labels = self.data[labels_name]
		if self.counter >= triplets.shape[0]:
			return None, None
		new_batch = triplets[self.counter:self.counter+batch_size, :, :], labels[self.counter:self.counter+batch_size, :]
		self.counter+=batch_size
		return new_batch

	def reset_reader(self):
		self.counter = 0

	def read_complete_set(self, triplets_name, labels_name,):
		return self.data[triplets_name], self.data[labels_name]

