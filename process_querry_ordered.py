import tensorflow as tf
import cPickle as pickle
import numpy as np
from triplet_reader import DataReader
#from concept2vec import concept_vector_model
import ncr_cnn_model
import sys
import train_oe
import ordered_embeding
import reader

def tokenize(phrase):
	return phrase.lower().replace(',',' ').replace('-',' ').replace(';', ' ').strip().split()	

def embed_phrase(phrase, wordList, padded_size):
	words = tokenize(phrase)
	em_list = [wordList[w] for w in words]
	em_list += [np.zeros(wordList["the"].shape)]*(padded_size-len(words))
	embeding = np.concatenate(em_list)
	return embeding

def check_phrase(phrase, wordList, word_limit):	
	words = tokenize(phrase)
	if len(words) <= word_limit and all([(w in wordList) for w in words]):
		return True
	return False



class NeuralAnnotator:

	def get_hp_id(self, querry):
		#if not check_phrase(querry, self.wordList, self.modelConfig.max_num_of_words):
		#	return None, None
		inp = self.rd.create_test_sample([querry])
		res = self.sess.run(self.model.get_querry_order_distance(), feed_dict = {self.model.input_sequence : inp[0], self.model.input_sequence_lengths: inp[1]})
		indecies = np.argsort(res[0,:])
		
		for i in range(5):
			print "------------------------------------------"

		num_printed = 0
		for i in indecies:
			print self.rd.concepts[i], self.rd.names[self.rd.concepts[i]], res[0,i]
			num_printed += 1
			if res[0,i] >= 1.0: # and num_printed>10:
				break


		
		
	def __init__(self):

		oboFile = open("hp.obo")
		vectorFile = open("vectors.txt")
		#vectorFile = open("train_data_gen/test_vectors.txt")

		self.rd = reader.Reader(oboFile, vectorFile)
		newConfig = train_oe.newConfig
		newConfig.vocab_size = self.rd.word2vec.shape[0]
		newConfig.word_embed_size = self.rd.word2vec.shape[1]
		newConfig.max_sequence_length = self.rd.max_length
		newConfig.hpo_size = len(self.rd.concept2id)

		self.model = ordered_embeding.NCRModel(newConfig)


		#init_op=tf.initialize_all_variables()
		saver = tf.train.Saver()
		self.sess = tf.Session()
		#sess.run(init_op)
		saver.restore(self.sess, 'checkpoints/training.ckpt')


	
def main():
	ant = NeuralAnnotator()

	while True:
		line = sys.stdin.readline().strip()
		ant.get_hp_id(line)
		
		if line == '':
			break


if __name__ == '__main__':
	main()

