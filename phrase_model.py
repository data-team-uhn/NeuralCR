import tensorflow as tf
import numpy as np


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def linear(x, shape):
	w = weight_variable(shape)
	b = weight_variable([shape[1]])
	return tf.matmul(x,w) + b


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
#		return tf.nn.rnn(cell, inputs, tf.gather(self.type_embedding, self.input_type_id), dtype=tf.float32, sequence_length=seq_length)

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
	def get_loss_phrase(self, embedding):
		### Lookup table HPO embedding ###
		input_HPO_embedding = self.get_HPO_embedding(self.input_hpo_id)
		p2c_loss = self.euclid_dis(embedding, input_HPO_embedding)
		p2c_loss += self.order_dis(embedding, input_HPO_embedding)
		return p2c_loss

	def get_loss_def(self, embedding):
		### Lookup table HPO embedding ###
		input_HPO_embedding = self.get_HPO_embedding(self.input_hpo_id)
		p2c_loss = self.euclid_dis(embedding, input_HPO_embedding)
		return p2c_loss


	def create_loss_var(self):
		print "set loss"
		input_HPO_embedding = self.get_HPO_embedding(self.input_hpo_id)

		cdistance = tf.transpose(self.order_dis_cartesian(input_HPO_embedding, self.get_HPO_embedding()))
		mask= tf.gather(self.ancestry_masks, self.input_hpo_id)

		c2c_pos = tf.reduce_sum(mask * cdistance, 1)
		c2c_neg = tf.reduce_sum((1-mask)*tf.maximum(0.0, self.config.alpha - cdistance), 1)
		c2c_loss = c2c_pos + c2c_neg

		general_loss = c2c_loss
		gru_loss = self.get_loss_phrase(self.gru_state)
#		gru_loss = tf.select(tf.equal(self.input_type_id, tf.zeros_like(self.input_sequence_lengths)), self.get_loss_phrase(self.gru_state), self.get_loss_def(self.gru_state))

		self.loss = tf.reduce_mean(general_loss + gru_loss)

	#############################
	##### Creates the model #####
	#############################
	def __init__(self, config, training = False):

		print "init"
		### Global Variables ###
		self.config = config

		self.HPO_embedding = embedding_variable("hpo_embedding", [config.hpo_size, config.hidden_size]) 
#		self.type_embedding = embedding_variable("type_embedding", [config.n_types, config.hidden_size]) 
		self.word_embedding = tf.get_variable("word_embedding", [config.vocab_size, config.word_embed_size])
		self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
		########################

		### Inputs ###
		self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
		self.input_hpo_id_unique = tf.placeholder(tf.int32, shape=[None])
		self.input_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		self.input_type_id = tf.placeholder_with_default(tf.zeros_like(self.input_sequence_lengths), shape=[None])

		#self.set_loss_for_input = tf.placeholder(tf.bool, shape=[])
		##############

		### Inputs ###
		self.def_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.def_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		#self.set_loss_for_def = tf.placeholder(tf.bool, shape=[])
		##############

		### Sequence prep & RNN ###
#		with tf.variable_scope("rnn0"):
#		_, gru_state_tmp = self.apply_rnn(self.input_sequence, self.input_sequence_lengths) 
		_, self.gru_state = self.apply_rnn(self.input_sequence, self.input_sequence_lengths) 
#		self.gru_state = tf.nn.sigmoid(linear(gru_state_tmp, [config.hidden_size, config.hidden_size]))
		#with tf.variable_scope("rnn1"):
		#	_, self.gru_state_rnn1 = self.apply_rnn(self.input_sequence, self.input_sequence_lengths) 
#		self.gru_state = tf.select(tf.equal(self.input_type_id, tf.zeros_like(self.input_sequence_lengths)), gru_state_rnn0, gru_state_rnn1)
		###########################

		if training:
			self.create_loss_var()

