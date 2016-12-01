import tensorflow as tf
import numpy as np


def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

'''
def weight_variable(name, shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
'''

def linear(name, x, shape):
	w = weight_variable(name+"W", shape)
	b = weight_variable(name+"B", [shape[1]])
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
		#		seq_embeded = tf.nn.embedding_lookup(self.word_embedding, seq)
#		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.config.max_sequence_length, seq_embeded)]
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.config.max_sequence_length, seq)]
		#'''
		w1 = weight_variable('layer1W', [self.config.word_embed_size, self.config.l1_size])
		b1 = weight_variable('layer1B', [self.config.l1_size])
		w2 = weight_variable('layer2W', [self.config.l1_size, self.config.l2_size])
		b2 = weight_variable('layer2B', [self.config.l2_size])
		'''
		w3 = weight_variable('layer3W', [l3_size, l3_size])
		b3 = weight_variable('layer3B', [l3_size])
		'''

		mlp_inputs = [tf.nn.tanh(tf.matmul(x, w1)+b1) for x in inputs]
		mlp_inputs = [tf.nn.tanh(tf.matmul(x, w2)+b2) for x in mlp_inputs]
		#mlp_inputs = [tf.nn.tanh(tf.matmul(x, w3)+b3) for x in mlp_inputs]
		#'''
		cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)

#		_, state = tf.nn.rnn(cell, inputs, dtype=tf.float32, sequence_length=seq_length)
		_, state = tf.nn.rnn(cell, mlp_inputs, dtype=tf.float32, sequence_length=seq_length)
		return state

		layer1 = tf.nn.softplus(linear('layer1', state, [self.config.hidden_size, self.config.hidden_size]))
		layer2 = tf.nn.softplus(linear('layer2', state, [self.config.hidden_size, self.config.hidden_size]))
		return tf.nn.tanh(linear('layer3', state, [self.config.hidden_size, self.config.concept_size]))

		return #		return tf.nn.rnn(cell, inputs, tf.gather(self.type_embedding, self.input_type_id), dtype=tf.float32, sequence_length=seq_length)

	## this would be zero if v is a child of u
	def order_dis(self, v, u):
		dif = u - v
		return tf.reduce_sum(tf.pow(tf.maximum(dif, tf.constant(0.0)), tf.constant(2.0)), reduction_indices=1) 

	def order_dis_cartesian(self, v, u):
		return tf.transpose(tf.map_fn(lambda x: self.order_dis(x,u), v, swap_memory=True))

	def order_dis_cartesian_trans(self, v, u):
		return tf.map_fn(lambda x: self.order_dis(v,x), u, swap_memory=True)

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
#		p2c_loss += self.order_dis(input_HPO_embedding, embedding)
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
		#gru_loss = self.create_new_loss_var() #self.get_loss_phrase(self.gru_state)
#		gru_loss = tf.select(tf.equal(self.input_type_id, tf.zeros_like(self.input_sequence_lengths)), self.get_loss_phrase(self.gru_state), self.get_loss_def(self.gru_state))

		self.loss = tf.reduce_mean(general_loss + gru_loss)


	def create_new_loss_var(self):
		print "set loss"
		
		input_HPO_embedding = self.get_HPO_embedding(self.input_hpo_id)

		cdistance = self.order_dis_cartesian_trans(self.get_HPO_embedding(), self.gru_state)

		mask= tf.gather(self.descendancy_masks, self.input_hpo_id)

		c2c_pos = tf.reduce_sum(mask * cdistance, 1)
		c2c_neg = tf.reduce_sum((1-mask)*tf.maximum(0.0, self.config.alpha - cdistance), 1)
		c2c_loss = c2c_pos + c2c_neg
		return c2c_loss


		self.loss = tf.reduce_mean(c2c_loss + p2c_loss)
		#self.loss = tf.reduce_mean(c2c_loss + p2c_loss)

	def mean_pool(self, seq, lens):
		#		return self.apply_rnn(seq, lens)
		seq_embeded = tf.nn.embedding_lookup(self.word_embedding, seq)
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.config.max_sequence_length, seq_embeded)]

		W1 = weight_variable('MeanW1', [self.config.word_embed_size, self.config.word_embed_size])
		B1 = weight_variable('MeanB1', [self.config.word_embed_size])
		layer1 = [tf.nn.sigmoid(tf.matmul(v, W1) + B1) for v in inputs]

		W2 = weight_variable('MeanW2', [self.config.word_embed_size, 1])
		B2 = weight_variable('MeanB2', [1])
		layer2 = [tf.nn.sigmoid(tf.matmul(v, W2) + B2) for v in layer1]

		weights = []
		weighted_inputs = []
		for i,v in enumerate(layer2):
			weights.append(tf.select(tf.less(i,lens), v, 0.0*v))
			weighted_inputs.append(inputs[i] * weights[i])


		mean = tf.add_n(weighted_inputs) #/ tf.add_n(weights)
		mean = tf.add_n(inputs) #/ tf.add_n(weights)

		fc1 = tf.nn.tanh(linear('fc1', mean, [self.config.word_embed_size, self.config.hidden_size]))
		fc2 = tf.nn.tanh(linear('fc2', fc1, [self.config.hidden_size, self.config.hidden_size]))
		fc3 = tf.nn.tanh(linear('fc3', fc2, [self.config.hidden_size, self.config.concept_size]))
		return fc3

	#############################
	##### Creates the model #####
	#############################
	def __init__(self, config, training = False):

		print "init"
		### Global Variables ###
		self.config = config

		self.HPO_embedding = embedding_variable("hpo_embedding", [config.hpo_size, config.concept_size]) 
#		self.type_embedding = embedding_variable("type_embedding", [config.n_types, config.hidden_size]) 
#		self.word_embedding = tf.get_variable("word_embedding", [config.vocab_size, config.word_embed_size], trainable=False)
		self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
		self.descendancy_masks = tf.transpose(self.ancestry_masks) # tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
		########################

		### Inputs ###
		self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
		self.input_hpo_id_unique = tf.placeholder(tf.int32, shape=[None])
		self.input_sequence = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, config.word_embed_size])
		#self.input_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		self.input_type_id = tf.placeholder_with_default(tf.zeros_like(self.input_sequence_lengths), shape=[None])

		#self.set_loss_for_input = tf.placeholder(tf.bool, shape=[])
		##############

		### Inputs ###
		self.def_sequence = tf.placeholder(tf.int32, shape=[None, config.max_sequence_length])
		self.def_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
		#self.set_loss_for_def = tf.placeholder(tf.bool, shape=[])
		##############

#		self.gru_state = self.mean_pool(self.input_sequence, self.input_sequence_lengths) 
		self.gru_state = self.apply_rnn(self.input_sequence, self.input_sequence_lengths) 

		### Sequence prep & RNN ###
#		with tf.variable_scope("rnn0"):
#		_, gru_state_tmp = self.apply_rnn(self.input_sequence, self.input_sequence_lengths) 
#		_, self.gru_state = self.apply_rnn(self.input_sequence, self.input_sequence_lengths) 
#		self.gru_state = self.mean_pool(self.input_sequence, self.input_sequence_lengths)
#		self.gru_state = tf.nn.sigmoid(linear(gru_state_tmp, [config.hidden_size, config.hidden_size]))
		#with tf.variable_scope("rnn1"):
		#	_, self.gru_state_rnn1 = self.apply_rnn(self.input_sequence, self.input_sequence_lengths) 
#		self.gru_state = tf.select(tf.equal(self.input_type_id, tf.zeros_like(self.input_sequence_lengths)), gru_state_rnn0, gru_state_rnn1)
		###########################

		if training:
			self.create_loss_var()
#			self.create_new_loss_var()

