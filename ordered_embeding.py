import tensorflow as tf
import numpy as np


def _embedding_variable(name, shape):
	return _weight_variable(name, shape)
	#	return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))

def _weight_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.constant_initializer(0.1))

def _linear(name, inp, shape):
	weights = _weight_variable(name+"_weights", shape)
	bias = _bias_variable(name+"_bias", [shape[1]])
	return tf.matmul(inp,weights)+bias

def multilayer(name, inp, shape, activation):
	tmp = inp
	for i in range(len(shape)-1):
		tmp = activation(_linear(name+"_layer_"+str(i), tmp, [shape[i],shape[i+1]]))
	return tmp




class NCRModel():
	def get_order_distance(self, reference_embedding, input_embedding):
		difs = reference_embedding - input_embedding
		#difs = tf.expand_dims(self.HPO_embedding, 0) - tf.expand_dims(input_embedding, 1)
		order_distance = tf.reduce_sum(tf.pow( tf.maximum(difs, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=2) 
		return order_distance

	def get_vector_distance(self, reference_embedding, input_embedding):
		difs = reference_embedding - input_embedding
		#difs = tf.expand_dims(self.HPO_embedding, 0) - tf.expand_dims(input_embedding, 1)
		order_distance = tf.reduce_sum(tf.pow( difs, tf.constant(2.0)), reduction_indices=2)
		return order_distance

	def get_querry_order_distance(self):
		difs = tf.expand_dims(self.HPO_embedding, 0) - tf.expand_dims(self.state, 1)
		#order_distance = tf.reduce_sum(tf.pow(difs, tf.constant(2.0)), reduction_indices=2) 
		order_distance = tf.reduce_sum(tf.pow( tf.maximum(difs, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=2) 
		return order_distance
		
	def get_pos_neg_loss(self, mask, distance, pos_only=False):
		positive_penalties = mask  * distance
		if pos_only:
			negative_penalties = 0
		else:
			negative_penalties = (1-mask)  * tf.maximum( self.alpha - distance, 0.0)
		penalties = positive_penalties + negative_penalties
		return tf.reduce_sum(penalties, [1])

		
	def __init__(self, config):
		self.alpha = config.alpha
		self.HPO_embedding = _embedding_variable("hpo_embedding", [config.hpo_size, config.hidden_size]) 

		print self.HPO_embedding.get_shape()
		self.word_embedding = tf.get_variable("word_embedding", [config.vocab_size, config.word_embed_size])
		#self.word_embedding = tf.get_variable("word_embedding", [config.vocab_size, config.word_embed_size], trainable = False)
		self.input_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
		#self.input_ancestry_mask = tf.placeholder(tf.float32, shape=[None,config.hpo_size])
		self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
		input_sequence_embeded = tf.nn.embedding_lookup(self.word_embedding, self.input_sequence)

		self.input_comp = tf.placeholder(tf.int32) #, shape=[None, config.comp_size])
		self.input_comp_mask = tf.placeholder(tf.float32) #, shape=[None, config.comp_size])

		'''
		with tf.variable_scope('pass1'):
			single_cell_p1 = tf.nn.rnn_cell.GRUCell(config.hidden_size)
		with tf.variable_scope('pass2'):
			single_cell_p2 = tf.nn.rnn_cell.GRUCell(config.hidden_size)
#		single_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_size)
		print config.hidden_size
		with tf.variable_scope('pass1'):
			cell_p1 = single_cell_p1
		with tf.variable_scope('pass2'):
			cell_p2 = single_cell_p2
		if config.num_layers > 1:
			with tf.variable_scope('pass1'):
				cell_p1 = tf.nn.rnn_cell.MultiRNNCell([single_cell_p1] * config.num_layers)
			with tf.variable_scope('pass2'):
				cell_p2 = tf.nn.rnn_cell.MultiRNNCell([single_cell_p2] * config.num_layers)
		'''

		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sequence_length, input_sequence_embeded)]

		initializer = tf.random_uniform_initializer(-0.01, 0.01 )
		with tf.variable_scope('forward'):
			cell_fw = tf.nn.rnn_cell.GRUCell(config.hidden_size/2)
		with tf.variable_scope('backward'):
			cell_bw = tf.nn.rnn_cell.GRUCell(config.hidden_size/2)

		self.outputs, self.state_fw, self.state_bw = tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		self.split_outputs = [tf.split(1,2,output) for output in self.outputs]
	#	self.densed_outputs = [out[0] for out in self.outputs]
		self.densed_outputs = self.outputs 
		'''
		self.outputs_fw, self.state_fw = tf.nn.rnn(cell_fw, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		self.outputs_bw_rev, self.state_bw = tf.nn.rnn(cell_bw, _reverse_seq(inputs, self.input_sequence_lengths), dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		self.outputs_bw = _reverse_seq(self.outputs_bw_rev, self.input_sequence_lengths)
		#self.outputs, self.state_fw = tf.nn.rnn(cell_fw, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		'''
		
		reference_embedding = tf.gather(self.HPO_embedding, self.input_comp) #tf.expand_dims(self.HPO_embedding, 0)
		self.distances = [ self.get_order_distance(reference_embedding, tf.expand_dims(input_embedding,1)) for input_embedding in self.densed_outputs]
		self.querry_distances = [ self.get_vector_distance(reference_embedding, tf.expand_dims(input_embedding,1)) for input_embedding in self.densed_outputs]

		self.final_distance = self.distances[0]
		for i,p in enumerate(self.distances):
			condition = (i+1 < self.input_sequence_lengths)
			self.final_distance = tf.select(condition, tf.minimum(self.final_distance, p), self.final_distance)
		self.querry_distance = self.querry_distances[0]
		for i,p in enumerate(self.querry_distances):
			condition = (i+1 < self.input_sequence_lengths)
			self.querry_distance = tf.select(condition, tf.minimum(self.querry_distance, p), self.querry_distance)

		input_loss = self.get_pos_neg_loss(self.input_comp_mask, self.final_distance)  
		input_id_hpo_embedding = tf.gather(self.HPO_embedding, self.input_hpo_id)
		hpo_loss = tf.select(tf.reduce_sum(self.input_comp_mask, 1)>0,
				self.get_pos_neg_loss(tf.gather(self.ancestry_masks, self.input_hpo_id),	self.get_order_distance(tf.expand_dims(self.HPO_embedding,0), tf.expand_dims(input_id_hpo_embedding,1))),
					tf.zeros_like(input_loss))
		self.new_loss =  tf.reduce_mean(input_loss + hpo_loss, [0]) 
						
		'''
		positive_penalties = self.input_comp_mask  * self.final_distance
		negative_penalties = (1-self.input_comp_mask)  * tf.maximum( self.alpha - self.final_distance, 0.0)
		penalties = positive_penalties + negative_penalties
				
		self.new_loss = tf.reduce_sum(penalties, [0,1]) + self.get_input_loss(input_ancestry_mask, tf.gather(self.HPO_embedding, self.input_hpo_id)
		'''


		'''
		cell = tf.nn.rnn_cell.GRUCell(config.hidden_size)
		self.outputs, self.state = tf.nn.rnn(cell, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		'''

		'''
		with tf.variable_scope('pass1'):
			outputs1, state_p1 = tf.nn.rnn(cell_p1,  inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		with tf.variable_scope('pass2'):
			outputs2, state_p2 = tf.nn.rnn(cell_p2,  inputs, initial_state = state_p1, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		self.state = state_p2

		self.new_loss = self.get_total_loss()
		self.final_distance = self.get_querry_order_distance()
		'''


