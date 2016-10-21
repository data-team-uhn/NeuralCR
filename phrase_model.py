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

	def apply_rnn(self, seq, seq_length):
		seq_embeded = tf.nn.embedding_lookup(self.word_embedding, seq)
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.config.max_sequence_length, seq_embeded)]
		cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
		return tf.nn.rnn(cell, inputs, dtype=tf.float32, sequence_length=seq_length)

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
	def get_loss(self, embedding):
		### Lookup table HPO embedding ###
		input_HPO_embedding = self.get_HPO_embedding(self.input_hpo_id)

		cdistance = tf.transpose(self.order_dis_cartesian(input_HPO_embedding, self.get_HPO_embedding()))
		mask= tf.gather(self.ancestry_masks, self.input_hpo_id)

		c2c_pos = tf.reduce_sum(mask * cdistance, 1)
		c2c_neg = tf.reduce_sum((1-mask)*tf.maximum(0.0, self.config.alpha - cdistance), 1)
		c2c_loss = c2c_pos + c2c_neg

		p2c_loss = self.euclid_dis(embedding, input_HPO_embedding)
		p2c_order_loss = self.order_dis(embedding, input_HPO_embedding)
		p2c_loss += p2c_order_loss

		return c2c_loss + p2c_loss
		#return tf.reduce_sum(c2c_loss + p2c_loss)
		
		'''
		NULL_concept_dis =  tf.transpose(self.order_dis_cartesian(self.gru_state, self.get_HPO_embedding()))
		NULL_concept_loss = tf.reduce_sum(tf.maximum(0.0, self.config.alpha - NULL_concept_dis), 1)
		self.loss = tf.reduce_mean(tf.select(tf.equal(self.input_hpo_id, self.config.concept_NULL), NULL_concept_loss, c2c_loss + p2c_loss))
		'''


	def create_loss_var(self):
		print "set loss"
		## if type A exists
		self.input_losses = self.get_loss(self.gru_state)
#		self.def_losses = self.get_loss(self.def_state)
		return

		'''
		
		total_loss = tf.cond(self.set_loss_for_input, lambda: self.get_loss(self.gru_state), lambda: tf.constant(0.0))
		self.loss = total_loss #self.get_loss(self.gru_state)
		return
		count_batch = tf.cond(self.set_loss_for_input, lambda: tf.shape(self.gru_state)[0], lambda: tf.constant(0))
		## if type B exists
		total_loss = tf.cond(self.set_loss_for_def, lambda: self.get_loss(self.def_state)+total_loss, lambda: total_loss)
		count_batch = tf.cond(self.set_loss_for_def, lambda: tf.shape(self.def_state)[0]+count_batch, lambda: count_batch)

		self.loss = total_loss / tf.to_float(count_batch)
		'''

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
		self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
		self.input_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		#self.set_loss_for_input = tf.placeholder(tf.bool, shape=[])
		##############

		### Inputs ###
		self.def_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.def_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		#self.set_loss_for_def = tf.placeholder(tf.bool, shape=[])
		##############


		### Sequence prep & RNN ###
		with tf.variable_scope("input-seq") as scope:
			self.gru_outputs, self.gru_state = self.apply_rnn(self.input_sequence, self.input_sequence_lengths) 
		###########################

		### Definition Sequence prep & RNN ###
		with tf.variable_scope("def-seq") as scope:
			self.def_outputs, self.def_state = self.apply_rnn(self.def_sequence, self.def_sequence_lengths) 
		###########################

		if training:
			self.create_loss_var()

