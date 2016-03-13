import tensorflow as tf
import numpy as np

def conv2d(x, w):
	return tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")

def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def conv_layer(x, filter_shape):
	conv_w = weight_variable(filter_shape)
	conv_bias = bias_variable( [filter_shape[3]] )

	conv = tf.nn.relu(conv2d(x, conv_w) + conv_bias)
	pool = pool2d(conv)

	return pool

def conv_pool_layer(x, filter_shape):
	conv_w = weight_variable(filter_shape)
	conv_bias = bias_variable( [filter_shape[3]] )

	conv = tf.nn.relu(conv2d(x, conv_w) + conv_bias)
	pool = tf.nn.max_pool(conv, [1, 10, 1, 1], [1, 10, 1, 1], "SAME")

	return pool

def fully_connected_layer(x, in_size, out_size):
	dense_weights=weight_variable([in_size,out_size])
	dense_bias=bias_variable([out_size])

	return tf.matmul(x,dense_weights) + dense_bias

def normalize(x):
	return x / tf.sqrt( tf.reduce_sum( tf.square(x), 1, keep_dims=True ))

def concept_vector_model(x, keep_prob):
	layer_1gram = conv_pool_layer(x, [1, 1, 100, 35])
	layer_2gram = conv_pool_layer(x, [2, 1, 100, 30])
	layer_3gram = conv_pool_layer(x, [3, 1, 100, 20])
	layer_4gram = conv_pool_layer(x, [4, 1, 100, 15])

	full_layer = tf.concat(3, [layer_1gram, layer_2gram, layer_3gram, layer_4gram])
	full_layer_dropout = tf.nn.dropout(full_layer,keep_prob)

	dense1 =  tf.nn.relu( fully_connected_layer(tf.reshape(full_layer_dropout, [-1, 100]), 100, 200) )

	return normalize(dense1)

