import tensorflow as tf
import cPickle as pickle
import numpy as np
from triplet_reader import DataReader
from concept2vec import concept_vector_model


#def run_epoch(session, 

def main():

	word_size = 100
	num_of_words = 10
	batch_size = 50

	data = pickle.load(open('triplets','rb'))
	#data = pickle.load(open('triplets_small','rb'))
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
	
		y_ = tf.concat(1, [tf.reduce_sum (g_v * c0_v, 1, keep_dims=True), tf.reduce_sum (g_v * c1_v, 1, keep_dims=True)])
		y_normalised = y_ / tf.reduce_sum(y_, 1, True)

		y = tf.placeholder(tf.float32, [None, 2])

		cross_entropy = - tf.reduce_sum(y * tf.log(y_normalised))

		train_step = tf.train.AdamOptimizer(2e-5).minimize(cross_entropy)
		
		correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_normalised,1))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init_op=tf.initialize_all_variables()

	merged = tf.merge_all_summaries()

	writer_i=0
	with tf.Session() as sess:
		writer = tf.train.SummaryWriter("/u/arbabi/ncr_tmp",
                                sess.graph.as_graph_def(add_shapes=True))
		sess.run(init_op)
		validation_set = reader.get_validation()
		test_set = reader.get_test()
		for epoch in range(20):
			reader.reset_reader()
			step=0
			while True:
				new_batch = reader.read_batch(batch_size)
				if new_batch == None:
					break
				if step%100 == 0:
					res = sess.run(accuracy, feed_dict={guide:new_batch[0], concept0:new_batch[1], concept1:new_batch[2], y:new_batch[3], keep_prob:1.0})
					#res = sess.run([merged, accuracy], feed_dict={guide:new_batch[0], concept0:new_batch[1], concept1:new_batch[2], y:new_batch[3], keep_prob:1.0})
					print "Step:: ", step, "Accuracy:: ", res
					#writer.add_summary(summary_str, writer_i)
				sess.run(train_step, feed_dict={guide:new_batch[0], concept0:new_batch[1], concept1:new_batch[2], y:new_batch[3], keep_prob:0.7})
				step+=1
				writer_i+=1
			
			print "Epoch:: ", epoch, "Validation Accuracy:: ", sess.run(accuracy, feed_dict={guide:validation_set[0], concept0:validation_set[1], concept1:validation_set[2], y:validation_set[3], keep_prob:1.0})

		print "Test Accuracy:: ", sess.run(accuracy, feed_dict={guide:test_set[0], concept0:test_set[1], concept1:test_set[2], y:test_set[3], keep_prob:1.0})

if __name__ == "__main__":
	main()

