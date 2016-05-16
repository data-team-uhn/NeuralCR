import tensorflow as tf
import cPickle as pickle
import numpy as np
from triplet_reader import DataReader
# from concept2vec import concept_vector_model
import ordered_embeding
import reader
import sys


'''
def read_files():
	directory="./train_data_gen/data_files/"
	files = ['validation_synonym_3_5_triplets', 'validation_synonym_3_5_labels', 'validation_synonym_2_2_triplets',
			 'validation_synonym_2_2_labels', 'validation_synonym_1_1_triplets', 'validation_synonym_1_1_labels',
			 'test_synonym_3_5_triplets', 'test_synonym_3_5_labels', 'test_synonym_2_2_triplets',
			 'test_synonym_2_2_labels', 'test_synonym_1_1_triplets', 'test_synonym_1_1_labels',
			 'training_triplets', 'training_labels']
	"""
	files = ['validation_synonym_3_5_triplets', 'validation_synonym_3_5_labels', 'validation_synonym_2_2_triplets',
			 'validation_synonym_2_2_labels', 'validation_synonym_1_1_triplets', 'validation_synonym_1_1_labels',
			 'validation_graph_3_5_triplets', 'validation_graph_3_5_labels', 'test_graph_2_2_triplets',
			 'test_graph_2_2_labels', 'test_synonym_3_5_triplets', 'test_synonym_3_5_labels',
			 'test_synonym_2_2_triplets', 'test_synonym_2_2_labels', 'test_synonym_1_1_triplets',
			 'test_synonym_1_1_labels', 'training_triplets', 'training_labels',
			 'test_graph_3_5_triplets', 'test_graph_3_5_labels', 'validation_graph_2_2_triplets',
			 'validation_graph_2_2_labels']
	"""
	data ={}
	for f in files:
		data[f] = np.load(directory+'/'+f+".npy")
	return data
	# validation_synonym_triplets = np.load(directory+'/validation_synonym_triplets.npy')

'''

class firstTrainConfig():
	lr_decay=0.8
	lr_init=1e-5
	batch_size = 100


class newConfig:
	hpo_size = 10000
	vocab_size = 50000
	hidden_size = 200
	word_embed_size = 100
	num_layers = 1
	max_sequence_length = 22
	alpha = 1

def run_epoch(sess, model, train_step, rd):
	rd.reset_counter()
	ii = 0
	while True:
		batch = rd.read_batch(50)
		if batch == None:
			break
		loss = sess.run(model.get_total_loss(), feed_dict = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_hpo_id:batch[2]})
		if ii % 20 == 0:
			print ii, loss
		sess.run(train_step, feed_dict = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_hpo_id:batch[2]})
		ii += 1

def traain():
	oboFile = open("hp.obo")
	vectorFile = open("vectors.txt")
	#vectorFile = open("train_data_gen/test_vectors.txt")

	rd = reader.Reader(oboFile, vectorFile)
	newConfig.vocab_size = rd.word2vec.shape[0]
	newConfig.word_embed_size = rd.word2vec.shape[1]
	newConfig.max_sequence_length = rd.max_length
	newConfig.hpo_size = len(rd.concept2id)

	model = ordered_embeding.NCRModel(newConfig)
	lr = tf.Variable(0.001, trainable=False)
	train_step = tf.train.AdamOptimizer(lr).minimize(model.get_total_loss())

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	sess.run(tf.assign(model.word_embedding, rd.word2vec))
	sess.run(tf.assign(model.ancestry_masks, rd.ancestry_mask))
	
#	print [batch[i].shape for i in range(3)]
#	loss = model.get_loss()
#	print sess.run(loss, feed_dict = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_ancestry_mask:batch[2]})
#	print sess.run(model.word_embedding, feed_dict = {model.input_vectors : batch[0], model.input_ancestry_mask:batch[1]}).shape
	saver = tf.train.Saver()

	lr_init = 0.01
	lr_decay = 0.8
	for epoch in range(100):
		print "epoch ::", epoch
		lr_new = lr_init * (lr_decay ** max(epoch-4.0, 0.0))
		sess.run(tf.assign(lr, lr_new))
		run_epoch(sess, model, train_step, rd)
		saver.save(sess, 'checkpoints/training.ckpt')

def main():
	traain()
#	tesst()

if __name__ == "__main__":
	main()

