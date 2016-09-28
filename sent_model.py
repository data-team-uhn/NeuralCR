import tensorflow as tf
import numpy as np

def embedding_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))

class NCRModel():

	def get_HPO_embedding(self, indices=None):
		embedding = self.HPO_embedding
		if indices is not None:
			embedding = tf.gather(self.HPO_embedding, indices)
		return embedding #tf.maximum(0.0, embedding)

	def apply_rnn(self, inputs):
		with tf.variable_scope('forward'):
			cell_fw = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
		with tf.variable_scope('backward'):
			cell_bw = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)

		tmp_outputs, gru_state_fw, self.gru_state_bw = tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)

		### Look ahead, then forget combination with a shift
		'''
		split_outputs = [tf.split(1,2,output) for output in tmp_outputs]
		self.gru_outputs = []
		for i in range(len(split_outputs)):
			if i+1<len(split_outputs):
				f = tf.sigmoid(tf.matmul(tf.concat(1, [split_outputs[i][0],split_outputs[i+1][1]]), self.rnnW)+self.rnnB)
				self.gru_outputs.append(f*split_outputs[i][0])
			else:
				f = tf.sigmoid(tf.matmul(tf.concat(1, [split_outputs[i][0],split_outputs[i][1]]), self.rnnW)+self.rnnB)
				self.gru_outputs.append(f*split_outputs[i][0])
		'''
		### Combine with a learned mixing coefficient
		split_outputs = [tf.split(1,2,output) for output in tmp_outputs]
		self.gru_outputs = []
		for i in range(len(split_outputs)):
			f = tf.sigmoid(tf.matmul(tmp_outputs[i], self.rnnW)+self.rnnB)
			self.gru_outputs.append(f*split_outputs[i][0]+(1-f)*split_outputs[i][1])

		### Look ahead, then forget combination
		'''
		split_outputs = [tf.split(1,2,output) for output in tmp_outputs]
		self.gru_outputs = []
		for i in range(len(split_outputs)):
			f = tf.sigmoid(tf.matmul(tmp_outputs[i], self.rnnW)+self.rnnB)
			self.gru_outputs.append(f*split_outputs[i][0])
		'''

		### Linear combination with a shift
		'''
		split_outputs = [tf.split(1,2,output) for output in tmp_outputs]
		self.gru_outputs = []
		for i in range(len(split_outputs)-1):
			self.gru_outputs.append(tf.tanh(tf.matmul(tf.concat(1, [split_outputs[i][0],split_outputs[i+1][1]]), self.rnnW)+self.rnnB))
		'''

		### Linear combination
		'''
		self.gru_outputs = [tf.tanh(tf.matmul(out, self.rnnW)+self.rnnB) for out in tmp_outputs]
		'''

		### Summation combination
		'''
		split_outputs = [tf.split(1,2,output) for output in tmp_outputs]
		self.gru_outputs = [out[0]+out[1] for out in split_outputs]
		'''

		### Summation combination with a shift
		'''
		self.gru_outputs = []
		for i in range(len(split_outputs)):
			if i+1<len(split_outputs):
				self.gru_outputs.append(split_outputs[i][0]+split_outputs[i+1][1])
			else:
				self.gru_outputs.append(split_outputs[i][0])
		'''

	def order_dis(self, v, u):
		dif = u - v
		return tf.reduce_sum(tf.pow(tf.maximum(dif, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=1) 

	def order_dis_cartesian(self, v, u):
		return tf.transpose(tf.map_fn(lambda x: self.order_dis(x,u), v, swap_memory=True))

	def euclid_dis(self, v ,u):
		return tf.reduce_sum(tf.pow(v-u, 2.0), 1)

	def euclid_dis_cartesian(self, v, u):
		return tf.reduce_sum(u*u, 1, keep_dims=True) + tf.expand_dims(tf.reduce_sum(v*v, 1), 0) - 2 * tf.matmul(u,v, transpose_b=True) 

	def rnn_minpool(self, v):
		distances = [self.euclid_dis(input_embedding, v) for input_embedding in self.gru_outputs]
		minpool = distances[0]
		for i,dis in enumerate(distances):
			condition = (i+1 < self.input_sequence_lengths)
			minpool = tf.select(condition, tf.minimum(minpool, dis), minpool)
		return minpool

	def rnn_minpool_cartesian(self, v):
		distances = [self.euclid_dis_cartesian(v, input_embedding) for input_embedding in self.gru_outputs]
		best_match_distance = distances[0]
		for i,dis in enumerate(distances):
			condition = (i+1 < self.input_sequence_lengths)
			condition_tiled = tf.tile(tf.expand_dims(condition,1), [1, tf.shape(v)[0]])
			best_match_distance = tf.select(condition_tiled, tf.minimum(best_match_distance, dis), best_match_distance)
		return best_match_distance


	#########################
	##### Loss Function #####
	#########################
	def set_loss(self):
		### Lookup table HPO embedding ###

		##
		'''
		neg_list = []
		neg_HPO_embedding = self.get_HPO_embedding(neg_list)
		distances = [self.euclid_dis_cartesian(input_embedding, neg_HPO_embedding) for input_embedding in self.gru_outputs]
		'''
		##
		input_HPO_embedding = self.get_HPO_embedding(self.input_hpo_id)
		pos_loss = self.rnn_minpool(input_HPO_embedding)
		self.loss = tf.reduce_mean(pos_loss)

	

	#############################
	##### Creates the model #####
	#############################
	def __init__(self, config, training = False):

		### Global Variables ###
		self.config = config

		self.HPO_embedding = tf.get_variable("hpo_embedding", [config.hpo_size, config.hidden_size], trainable=False) 
		self.word_embedding = tf.get_variable("word_embedding", [config.vocab_size, config.word_embed_size])
		#self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
		########################

		### Inputs ###
		self.input_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
		##############

		### Sequence prep & RNN ###
		input_sequence_embeded = tf.nn.embedding_lookup(self.word_embedding, self.input_sequence)
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sequence_length, input_sequence_embeded)]
		self.rnnW = tf.get_variable('rnnW', shape=[2*self.config.hidden_size, self.config.hidden_size], initializer = tf.random_normal_initializer(stddev=0.1))
		self.rnnB = tf.get_variable('rnnB', shape=[self.config.hidden_size], initializer = tf.random_normal_initializer(stddev=0.1))
		
		self.apply_rnn(inputs) 
		###########################

		if training:
			self.set_loss()

