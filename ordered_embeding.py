import tensorflow as tf
import numpy as np


def _embedding_variable(name, shape):
	return _weight_variable(name, shape)

def _weight_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))

def _bias_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.constant_initializer(0.1))

class NCRModel():

	def get_HPO_embedding(self, indices=None):
		embedding = self.HPO_embedding
		if indices is not None:
			embedding = tf.gather(self.HPO_embedding, indices)
		return embedding #tf.maximum(0.0, embedding)


	def apply_rnn(self, inputs, stemmed_inputs):
		if self.config.last_state:
			return self.apply_rnn_unidir(inputs, stemmed_inputs) 
		else:
			tmp_outputs, gru_state_fw, self.gru_state_bw = self.apply_rnn_bidir(inputs) 
			split_outputs = [tf.split(1,2,output) for output in tmp_outputs]
			gru_outputs = [out[0]+out[1] for out in split_outputs]
			return gru_outputs, None 

	def apply_rnn_unidir(self, inputs, stemmed_inputs):
		'''
		with tf.variable_scope("pass1"):
			cell_pass1 = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
			tmp_out, tmp_state = tf.nn.rnn(cell_pass1, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		with tf.variable_scope("pass2"):
			cell_pass2 = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
			return tf.nn.rnn(cell_pass2, inputs, tmp_state, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		'''

		cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
		mixed_input = [(v+u)/2.0 for v,u in zip(inputs, stemmed_inputs)]

		concat_input = [tf.pack([v,u], axis=2) for v,u in zip(inputs, stemmed_inputs)]
		a_weight = _weight_variable("q_weights", shape = [1, self.config.word_embed_size,1])
		logits =  [tf.nn.tanh(tf.reduce_sum(v * a_weight, [1])) for v in concat_input]
		alphas = [tf.nn.softmax(logit) for logit in logits]

#		mixed_input = [tf.reduce_sum(inp*tf.expand_dims(alphas[i],1), [2]) for i,inp in enumerate(concat_input)]

		##
		'''
		alpha = tf.nn.softmax
		mixed_input_by_ = [(v+u)/2.0 for v,u in zip(inputs, stemmed_inputs)]
		'''
		##
		#return tf.nn.rnn(cell, stemmed_inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		#return tf.nn.rnn(cell, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		return tf.nn.rnn(cell, mixed_input, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		#return tf.nn.rnn(cell, (inputs+stemmed_inputs)/2.0, dtype=tf.float32, sequence_length=self.input_sequence_lengths)

	def apply_rnn_bidir(self, inputs):
		'''
		with tf.variable_scope('forward'):
			cell_fw = tf.nn.rnn_cell.GRUCell(self.config.hidden_size/2, activation=tf.nn.tanh)
		with tf.variable_scope('backward'):
			cell_bw = tf.nn.rnn_cell.GRUCell(self.config.hidden_size/2, activation=tf.nn.tanh)
		'''
		with tf.variable_scope('forward'):
			cell_fw = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
		with tf.variable_scope('backward'):
			cell_bw = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
		return tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)

	def order_dis(self, v, u):
		dif = u - v
		return tf.reduce_sum(tf.pow(tf.maximum(dif, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=1) 

	def order_dis_cartesian(self, v, u):
		dif = tf.expand_dims(u,1) - tf.expand_dims(v,0)
		return tf.reduce_sum(tf.pow(tf.maximum(dif, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=2) 

	def euclid_dis(self, v ,u):
		return tf.reduce_sum(tf.pow(v-u, 2.0), 1)

	def euclid_dis_cartesian(self, v, u):
		dif = tf.expand_dims(u,1) - tf.expand_dims(v,0)
		return tf.reduce_sum(tf.pow(dif, 2.0), 2)


	#########################
	##### Loss Function #####
	#########################
	def get_loss(self):
		### Lookup table HPO embedding ###
		input_HPO_embedding = self.get_HPO_embedding(self.input_hpo_id)

		cdistance = tf.transpose(self.order_dis_cartesian(input_HPO_embedding, self.get_HPO_embedding()))
		mask= tf.gather(self.ancestry_masks, self.input_hpo_id)

		c2c_pos = tf.reduce_sum(mask * cdistance, 1)
		c2c_neg = tf.reduce_sum((1-mask)*tf.maximum(0.0, self.config.alpha - cdistance), 1)
		c2c_loss = c2c_pos + c2c_neg

		if self.config.last_state:
			p2c_loss = self.euclid_dis(self.gru_state, input_HPO_embedding)
			p2c_order_loss = self.order_dis(self.gru_state, input_HPO_embedding)
			p2c_loss += p2c_order_loss
		else:
			distances = [self.euclid_dis(input_embedding, input_HPO_embedding) for input_embedding in self.gru_outputs]
			p2c_loss = distances[0]
			for i,p in enumerate(distances):
				condition = (i+1 < self.input_sequence_lengths)
				p2c_loss = tf.select(condition, tf.minimum(p2c_loss, p), p2c_loss)
		
		return tf.reduce_mean(c2c_loss + p2c_loss)
	

	################################################################
	##### Returns distances between HPO concepts and the input #####
	################################################################
	def get_querry_dis(self):
		if self.config.last_state:
			return self.euclid_dis_cartesian(self.get_HPO_embedding(), self.gru_state)
		else:
			distances = [self.euclid_dis_cartesian(self.get_HPO_embedding(), input_embedding) for input_embedding in self.gru_outputs]
			final_dis = distances[0]
			for i,p in enumerate(distances):
				condition = (i+1 < self.input_sequence_lengths)
				final_dis = tf.select(condition, tf.minimum(final_dis, p), final_dis)
			return final_dis


	#############################
	##### Creates the model #####
	#############################
	def __init__(self, config):

		### Global Variables ###
		self.config = config

		self.HPO_embedding = _embedding_variable("hpo_embedding", [config.hpo_size, config.hidden_size]) 
		self.word_embedding = tf.get_variable("word_embedding", [config.vocab_size, config.word_embed_size])
		self.stemmed_word_embedding = tf.get_variable("stemmed_word_embedding", [config.stemmed_vocab_size, config.word_embed_size])
		self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
		########################

		### Inputs ###
		self.input_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.input_stemmed_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
		##############

		### Sequence prep & RNN ###
		input_sequence_embeded = tf.nn.embedding_lookup(self.word_embedding, self.input_sequence)
		input_stemmed_sequence_embeded = tf.nn.embedding_lookup(self.stemmed_word_embedding, self.input_stemmed_sequence)
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sequence_length, input_sequence_embeded)]
		stemmed_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sequence_length, input_stemmed_sequence_embeded)]
		
		self.gru_outputs, self.gru_state = self.apply_rnn(inputs, stemmed_inputs) 
		###########################


