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
		cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
		return tf.nn.rnn(cell, inputs, dtype=tf.float32, sequence_length=self.input_sequence_lengths)

	def order_dis(self, v, u):
		dif = u - v
		return tf.reduce_sum(tf.pow(tf.maximum(dif, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=1) 

	def order_dis_cartesian(self, v, u):
		return tf.transpose(tf.map_fn(lambda x: self.order_dis(x,u), v, swap_memory=True))

	def euclid_dis(self, v ,u):
		return tf.reduce_sum(tf.pow(v-u, 2.0), 1)

	def euclid_dis_cartesian(self, v, u):
		return tf.reduce_sum(u*u, 1, keep_dims=True) + tf.expand_dims(tf.reduce_sum(v*v, 1), 0) - 2 * tf.matmul(u,v, transpose_b=True) 

	#########################
	##### Loss Function #####
	#########################
	def set_loss(self):
		print "set loss"
		### Lookup table HPO embedding ###
		input_HPO_embedding = self.get_HPO_embedding(self.input_hpo_id)

		cdistance = tf.transpose(self.order_dis_cartesian(input_HPO_embedding, self.get_HPO_embedding()))
		mask= tf.gather(self.ancestry_masks, self.input_hpo_id)

		c2c_pos = tf.reduce_sum(mask * cdistance, 1)
		c2c_neg = tf.reduce_sum((1-mask)*tf.maximum(0.0, self.config.alpha - cdistance), 1)
		c2c_loss = c2c_pos + c2c_neg

		p2c_loss = self.euclid_dis(self.gru_state, input_HPO_embedding)
		p2c_order_loss = self.order_dis(self.gru_state, input_HPO_embedding)
		p2c_loss += p2c_order_loss
		
		self.loss = tf.reduce_mean(c2c_loss + p2c_loss)
	

	#############################
	##### Creates the model #####
	#############################
	def __init__(self, config, training = False):

		print "init"
		### Global Variables ###
		self.config = config

		self.HPO_embedding = embedding_variable("hpo_embedding", [config.hpo_size, config.hidden_size]) 
		self.word_embedding = tf.get_variable("word_embedding", [config.vocab_size, config.word_embed_size])
		self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
		########################

		### Inputs ###
		self.input_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
		##############

		### Sequence prep & RNN ###
		input_sequence_embeded = tf.nn.embedding_lookup(self.word_embedding, self.input_sequence)
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.max_sequence_length, input_sequence_embeded)]
		
		self.gru_outputs, self.gru_state = self.apply_rnn(inputs) 
		###########################

		if training:
			self.set_loss()

