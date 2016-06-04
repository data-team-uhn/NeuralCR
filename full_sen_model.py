import tensorflow as tf
import numpy as np


def _embedding_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(0.5, 0.25))
#return _weight_variable(name, shape)
	#	return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))

def _weight_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.constant_initializer(0.1))

class NCRModel():

	'''
	def get_order_distance(self, input_embedding):
		difs = tf.expand_dims(self.HPO_embedding, 0) - tf.expand_dims(input_embedding, 1)
		order_distance = tf.reduce_sum(tf.pow( tf.maximum(difs, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=2) 
		return order_distance

	def get_hpo_order_distance(self):
		difs = tf.expand_dims(self.HPO_embedding, 0) - tf.expand_dims(self.HPO_embedding, 1)
		order_distance = tf.reduce_sum(tf.pow( tf.maximum(difs, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=2) 
		return order_distance

	def get_querry_order_distance(self):
		difs = tf.expand_dims(self.HPO_embedding, 0) - tf.expand_dims(self.state, 1)
		#order_distance = tf.reduce_sum(tf.pow(difs, tf.constant(2.0)), reduction_indices=2) 
		order_distance = tf.reduce_sum(tf.pow( tf.maximum(difs, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=2) 
		return order_distance
		
	def get_input_loss(self, ancestry_mask, input_embedding):
		order_distance = self.get_order_distance(input_embedding)
		positive_penalties = ancestry_mask  * order_distance
		negative_penalties = (1-ancestry_mask)  * tf.maximum( self.alpha - order_distance, 0.0)
		penalties = positive_penalties + negative_penalties
		#penalties = (positive_penalties / tf.reduce_sum(self.input_ancestry_mask)
		#		+ negative_penalties / tf.reduce_sum(1 - self.input_ancestry_mask) )
				
		loss = tf.reduce_sum(penalties, [0,1])
		return loss 
	'''


	'''
	def get_input_loss(self, input_embedding, comp_embedding):
		order_distance = self.get_order_distance_from_comparables(input_embedding, comp_embedding)
		positive_penalties = self.input_comp_mask  * order_distance
		negative_penalties = (1-ancestry_mask)  * tf.maximum( self.alpha - order_distance, 0.0)
		penalties = positive_penalties + negative_penalties
		#penalties = (positive_penalties / tf.reduce_sum(self.input_ancestry_mask)
		#		+ negative_penalties / tf.reduce_sum(1 - self.input_ancestry_mask) )
				
		loss = tf.reduce_sum(penalties, [0,1])
		return loss 


	def get_total_loss(self):
		input_ancestry_mask = tf.gather(self.ancestry_masks, self.input_hpo_id)
		return self.get_input_loss(input_ancestry_mask, self.state_fw) + self.get_input_loss(input_ancestry_mask, tf.gather(self.HPO_embedding, self.input_hpo_id))
	'''

		
	def get_order_distance_from_comparables(self, input_embedding, comp_embedding):
		difs = comp_embedding - tf.expand_dims(input_embedding, 1)
		#difs = tf.expand_dims(self.comp_embedding, 0) - tf.expand_dims(input_embedding, 1)
		order_distance = tf.reduce_sum(tf.pow( tf.maximum(difs, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=2) 
		return order_distance

	def __init__(self, config):

	#	self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
		#self.input_ancestry_mask = tf.placeholder(tf.float32, shape=[None,config.hpo_size])

		self.alpha = config.alpha
		self.word_embedding = tf.get_variable("word_embedding", [config.vocab_size, config.word_embed_size])
		self.HPO_embedding = _embedding_variable("hpo_embedding", [config.hpo_size, config.hidden_size]) 

		self.input_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		input_sequence_embeded = tf.nn.embedding_lookup(self.word_embedding, self.input_sequence)

		self.input_comp = tf.placeholder(tf.int32) #, shape=[None, config.comp_size])
		self.input_comp_mask = tf.placeholder(tf.float32) #, shape=[None, config.comp_size])

#		self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sequence_length, input_sequence_embeded)]

		single_cell_fw = tf.nn.rnn_cell.GRUCell(config.hidden_size)
		single_cell_bw = tf.nn.rnn_cell.GRUCell(config.hidden_size)
		'''
		single_cell_fw = tf.nn.rnn_cell.LSTMCell(config.hidden_size)
		single_cell_bw = tf.nn.rnn_cell.LSTMCell(config.hidden_size)
		'''
		cell_fw = single_cell_fw
		cell_bw = single_cell_bw
		if config.num_layers > 1:
			cell_fw = tf.nn.rnn_cell.MultiRNNCell([single_cell_fw] * config.num_layers)
			cell_bw = tf.nn.rnn_cell.MultiRNNCell([single_cell_bw] * config.num_layers)

		'''
		self.outputs, self.state_fw, self.state_bw = tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)

		densed_weights = _weight_variable("densed_weights", [2*config.hidden_size, config.hidden_size])
		densed_bias = _bias_variable("densed_bias", [config.hidden_size])
		self.densed_outputs = [ tf.nn.relu(tf.matmul(rnn_output, densed_weights) + densed_bias) for rnn_output in self.outputs ]
		'''
		self.outputs, self.state_fw = tf.nn.rnn(cell_fw, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
	
		
		self.comp_embedding = tf.gather(self.HPO_embedding, self.input_comp)
		#self.distances = [ self.get_order_distance_from_comparables(input_embedding, self.comp_embedding) for input_embedding in self.densed_outputs]

		self.final_distance = self.get_order_distance_from_comparables(self.state_fw, self.comp_embedding)
		
		'''
		self.final_distance = self.distances[0]
		for p in self.distances:
			self.final_distance = tf.minimum(self.final_distance, p)
		'''

		positive_penalties = self.input_comp_mask  * self.final_distance
		negative_penalties = (1-self.input_comp_mask)  * tf.maximum( self.alpha - self.final_distance, 0.0)
		self.penalties = positive_penalties + negative_penalties
				
		self.new_loss = tf.reduce_sum(self.penalties, [0,1])


		'''
		hpo_expanded = tf.expand_dims(self.HPO_embedding, 0)
		self.diffs = [ tf.expand_dims(densed_output, 1) - hpo_expanded for densed_output in self.densed_outputs]
		reduce_weights = _weight_variable("reduce_weights", [config.hidden_size, 1])
		reduce_bias = _bias_variable("reduce_bias", [1])
		self.logits = [tf.reshape(tf.matmul(tf.reshape(diff, [-1,config.hidden_size]),reduce_weights) + reduce_bias, [-1,config.hpo_size]) for diff in self.diffs]
		self.final_logits = self.logits[0]
		for p in self.logits:
			self.final_logits = tf.maximum(self.final_logits, p)
		
		input_ancestry_mask = tf.gather(self.ancestry_masks, self.input_hpo_id)
		self.new_loss = tf.reduce_mean ( tf.reduce_sum ( tf.nn.sigmoid_cross_entropy_with_logits(self.final_logits, input_ancestry_mask), [1])) #+ self.get_input_loss(input_ancestry_mask, tf.gather(self.HPO_embedding, self.input_hpo_id))
		'''
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

		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sequence_length, input_sequence_embeded)]
#		self.outputs, self.state = tf.nn.rnn(cell, inputs, self.rnn_init_state, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		with tf.variable_scope('pass1'):
			outputs1, state_p1 = tf.nn.rnn(cell_p1,  inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		with tf.variable_scope('pass2'):
			outputs2, state_p2 = tf.nn.rnn(cell_p2,  inputs, initial_state = state_p1, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		self.state = state_p2
		'''

#		self.total_loss = self.get_total_loss()
		
