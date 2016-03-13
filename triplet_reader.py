import numpy as np

class DataReader:
	
	def __init__(self, data):
		total=len(data)
		self.training = data[:int(total*0.8)]
		self.validation = data[int(total*0.8):int(total*0.9)]
		self.test = data[int(total*0.9):]
		self.counter = 0

	def read_batch(self, batch_size):
		if self.counter >= len(self.training):
			return None
		new_batch = [np.stack( [triplet[i] for triplet in self.training[ self.counter : min(len(self.training), self.counter+batch_size)]]) for i in range(4)]
		self.counter+=batch_size
		return new_batch

	def reset_reader(self):
		self.counter = 0

	def get_validation(self):
		new_batch = [np.stack( [triplet[i] for triplet in self.validation]) for i in range(4)]
		return new_batch

	def get_test(self):
		new_batch = [np.stack( [triplet[i] for triplet in self.test]) for i in range(4)]
		return new_batch

