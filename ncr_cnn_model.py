import tensorflow as tf
import numpy as np

class bigConfig:
	max_num_of_words = 10
	word_size = 100
	conv_layer1_size=1024
	conv_layer2_size=2048
	conv_layer3_size=1024
	dense_layer1_size=1024
	dense_layer2_size=300
#	dense_layer3_size=500

class smallConfig:
	max_num_of_words = 10
	word_size = 100
	conv_layer1_size=256
	conv_layer2_size=512
	dense_layer1_size=1024
	dense_layer2_size=1024
	dense_layer3_size=250

def _conv2d(x, w):
	return tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")

def _weight_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1)) 

def _bias_variable(name, shape):
	return tf.get_variable(name, shape = shape, initializer = tf.constant_initializer(0.1)) 

def _conv_pool_layer(x, name, filter_shape):
	conv_w = _weight_variable(name+"weights", filter_shape)
	conv_bias = _bias_variable(name+"bias", [filter_shape[3]] )

	conv = tf.nn.relu(_conv2d(x, conv_w) + conv_bias)
#	pool = tf.nn.max_pool(conv, [1, 10, 1, 1], [1, 10, 1, 1], "SAME")

	return conv
	#return pool

def _fully_connected_layer(x, name, in_size, out_size):
	dense_weights=_weight_variable(name+"weights", [in_size,out_size])
#	tf.histogram_summary("dense weights", dense_weights)
	dense_bias=_bias_variable(name+"bias",[out_size])

	return tf.matmul(x,dense_weights) + dense_bias

def normalize(x):
	return x / tf.sqrt( tf.reduce_sum( tf.square(x), 1, keep_dims=True ))

class NCRModel():
	def __init__(self, config):
		self.input_vectors = tf.placeholder(tf.float32, [None, config.word_size * config.max_num_of_words])
		input_reshaped = tf.reshape(self.input_vectors, [-1, config.max_num_of_words, 1, config.word_size])
		layer1_1gram = _conv_pool_layer(input_reshaped, "layer1_1gram_", [1, 1, 100, config.conv_layer1_size])
		layer2_1gram = _conv_pool_layer(layer1_1gram, "layer2_1gram_", [1, 1, config.conv_layer1_size, config.conv_layer2_size])
		layer3_1gram = _conv_pool_layer(layer2_1gram, "layer3_1gram_", [1, 1, config.conv_layer2_size, config.conv_layer3_size])
		pooled_layer = tf.nn.max_pool(layer3_1gram, [1, 10, 1, 1], [1, 10, 1, 1], "SAME")

	#	full_layer = tf.concat(3, [pooled_layer]) #, layer_2gram, layer_3gram, layer_4gram, layer_5gram, layer_6gram])
		#full_layer = tf.concat(3, [layer_1gram, layer_2gram, layer_3gram, layer_4gram, layer_5gram, layer_6gram])
#		full_layer_dropout = tf.nn.dropout(pooled_layer,keep_prob)

		dense1 =  tf.nn.relu( _fully_connected_layer(tf.reshape(pooled_layer, [-1, config.conv_layer3_size]), "dense1_", config.conv_layer3_size, config.dense_layer1_size) )
		dense2 =  tf.nn.tanh( _fully_connected_layer(dense1, "dense2_", config.dense_layer1_size, config.dense_layer2_size) )
#		dense2 =  tf.nn.relu( _fully_connected_layer(dense1, "dense2_", config.dense_layer1_size, config.dense_layer2_size) )
#		dense3 = tf.nn.tanh( _fully_connected_layer(dense2, "dense3_", config.dense_layer2_size, config.dense_layer3_size))

		self.rep = normalize(dense2)

