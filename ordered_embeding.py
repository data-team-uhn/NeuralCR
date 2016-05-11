import tensorflow as tf
import numpy as np


def _embedding_variable(name, shape):
	return _weight_variable(name, shape)
	#	return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))

def _weight_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.constant_initializer(0.1))

class NCRModel():
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
		'''
		penalties = (positive_penalties / tf.reduce_sum(self.input_ancestry_mask)
				+ negative_penalties / tf.reduce_sum(1 - self.input_ancestry_mask) )
		'''
				
		loss = tf.reduce_sum(penalties, [0,1])
		return loss 

	def get_total_loss(self):
		input_ancestry_mask = tf.gather(self.ancestry_masks, self.input_hpo_id)
		return self.get_input_loss(input_ancestry_mask, self.state) + self.get_input_loss(input_ancestry_mask, tf.gather(self.HPO_embedding, self.input_hpo_id))

		
	def __init__(self, config):
		self.alpha = config.alpha
		self.HPO_embedding = _embedding_variable("hpo_embedding", [config.hpo_size, config.hidden_size]) 
		print self.HPO_embedding.get_shape()
		self.word_embedding = tf.get_variable("word_embedding", [config.vocab_size, config.word_embed_size])
		self.input_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
		#self.input_ancestry_mask = tf.placeholder(tf.float32, shape=[None,config.hpo_size])
		self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
		input_sequence_embeded = tf.nn.embedding_lookup(self.word_embedding, self.input_sequence)

		single_cell = tf.nn.rnn_cell.GRUCell(config.hidden_size)
#		single_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_size)
		print config.hidden_size
		cell = single_cell
		if config.num_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.num_layers)

		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sequence_length, input_sequence_embeded)]
		self.outputs, self.state = tf.nn.rnn(cell, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)
		
