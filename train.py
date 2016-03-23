import tensorflow as tf
import cPickle as pickle
import numpy as np
from triplet_reader import DataReader
from concept2vec import concept_vector_model

def read_files():
	directory="./train_data_gen/data_files/"
	files = ['validation_synonym_3_5_triplets','validation_synonym_3_5_labels', 'validation_synonym_2_2_triplets','validation_synonym_2_2_labels', 'validation_synonym_1_1_triplets', 'validation_synonym_1_1_labels', 'validation_graph_3_5_triplets', 'validation_graph_3_5_labels', 'test_graph_2_2_triplets', 'test_graph_2_2_labels', 'test_synonym_3_5_triplets', 'test_synonym_3_5_labels', 'test_synonym_2_2_triplets', 'test_synonym_2_2_labels', 'test_synonym_1_1_triplets', 'test_synonym_1_1_labels', 'training_triplets', 'training_labels', 'test_graph_3_5_triplets', 'test_graph_3_5_labels', 'validation_graph_2_2_triplets', 'validation_graph_2_2_labels']
	data ={}
	for f in files:
		data[f] = np.load(directory+'/'+f+".npy")
	return data
#	validation_synonym_triplets = np.load(directory+'/validation_synonym_triplets.npy')

class bigConfig:
	conv_layer1_size=1024
	conv_layer2_size=2048
	dense_layer1_size=4096
	dense_layer2_size=4096
	dense_layer3_size=500

class smallConfig:
	conv_layer1_size=256
	conv_layer2_size=512
	dense_layer1_size=1024
	dense_layer2_size=1024
	dense_layer3_size=250

def main():

	word_size = 100
	num_of_words = 10
	batch_size = 400
	config = smallConfig()
	data = read_files()	

	reader = DataReader(data)

	guide    = tf.placeholder(tf.float32, [None, word_size * num_of_words])
	concept0 = tf.placeholder(tf.float32, [None, word_size * num_of_words])
	concept1 = tf.placeholder(tf.float32, [None, word_size * num_of_words])
 
 	g = tf.reshape(guide, [-1, num_of_words, 1, word_size])
 	c0 = tf.reshape(concept0, [-1, num_of_words, 1, word_size])
 	c1 = tf.reshape(concept1, [-1, num_of_words, 1, word_size])

	keep_prob = tf.placeholder(tf.float32)

	with tf.variable_scope("models"):
		g_v = concept_vector_model(g, keep_prob)
	with tf.variable_scope("models", reuse=True):
		c0_v = concept_vector_model(c0, keep_prob)
		c1_v = concept_vector_model(c1, keep_prob)
#		tf.histogram_summary("dense weights", tf.get_variable( "dense_weights") )
	
	y_ = tf.concat(1, [tf.reduce_sum (g_v * c0_v, 1, keep_dims=True), tf.reduce_sum (g_v * c1_v, 1, keep_dims=True)])
	y_normalised = y_ / tf.reduce_sum(y_, 1, True)

	y = tf.placeholder(tf.float32, [None, 2])

	cross_entropy = - tf.reduce_sum(y * tf.log(y_normalised))

	lr_decay=0.8
	lr_init=1e-5
	lr = tf.Variable( 0.0, trainable=False)
	train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
	
	correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_normalised,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init_op=tf.initialize_all_variables()

	merged = tf.merge_all_summaries()

	validation_sets = {}
	test_sets = {}
	for val_set in ['validation_synonym_3_5', 'validation_synonym_2_2', 'validation_synonym_1_1', 'validation_graph_3_5', 'validation_graph_2_2']:
		validation_sets[val_set] = reader.read_complete_set(val_set+ '_triplets', val_set + '_labels')
	for test_set in ['test_synonym_3_5', 'test_synonym_2_2', 'test_synonym_1_1', 'test_graph_3_5', 'test_graph_2_2']:
		test_sets[test_set] = reader.read_complete_set(test_set+ '_triplets', test_set + '_labels')
	all_training = reader.read_complete_set('training_triplets', 'training_labels')

	writer_i=0
	with tf.Session() as sess:
		writer = tf.train.SummaryWriter("/u/arbabi/ncr_tmp",
                                sess.graph.as_graph_def(add_shapes=True))
		sess.run(init_op)

		for epoch in range(20):
			reader.reset_reader()
			step=0

			lr_new = lr_init * (lr_decay ** max(epoch-4.0, 0.0))
			sess.run(tf.assign(lr, lr_new))
			print "lr :: ", lr_new

			print "Epoch:: ", epoch 

			for val_set in validation_sets:
				the_set = validation_sets[val_set]
				print val_set + " Accuracy :: ", sess.run(accuracy, feed_dict={guide:the_set[0][:,:,0], concept0:the_set[0][:,:,1], concept1:the_set[0][:,:,2], y:the_set[1], keep_prob:1.0})
			print "Training Accuracy :: ", sess.run(accuracy, feed_dict={guide:the_set[0][:,:,0], concept0:the_set[0][:,:,1], concept1:the_set[0][:,:,2], y:the_set[1], keep_prob:1.0})

			while True:
				new_batch, labels = reader.read_batch('training_triplets', 'training_labels', batch_size)
				if new_batch == labels:
					break
				if step%100 == 0:
					res = sess.run(accuracy, feed_dict={guide:new_batch[:,:,0], concept0:new_batch[:,:,1], concept1:new_batch[:,:,2], y:labels, keep_prob:1.0})
					print "Step:: ", step, "Accuracy:: ", res
					#writer.add_summary(res[0], writer_i)
#				sess.run(train_step, feed_dict={guide:new_batch[0], concept0:new_batch[1], concept1:new_batch[2], y:new_batch[3], keep_prob:1.0})
				sess.run(train_step, feed_dict={guide:new_batch[:,:,0], concept0:new_batch[:,:,1], concept1:new_batch[:,:,2], y:labels, keep_prob:1.0})
				step+=1
				writer_i+=1

		for test_set in test_sets:
			the_set = test_sets[test_set]
			print test_set + " Accuracy :: ", sess.run(accuracy, feed_dict={guide:the_set[0][:,:,0], concept0:the_set[0][:,:,1], concept1:the_set[0][:,:,2], y:the_set[1], keep_prob:1.0})
		


if __name__ == "__main__":
	main()

