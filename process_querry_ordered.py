import tensorflow as tf
import cPickle as pickle
import numpy as np
from triplet_reader import DataReader
#from concept2vec import concept_vector_model
import ncr_cnn_model
import sys
import train_oe
from ordered_embeding import NCRModel
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

	def get_hp_id(self, querry, count):
		#if not check_phrase(querry, self.wordList, self.modelConfig.max_num_of_words):
		#	return None, None
		inp = self.rd.create_test_sample([querry])
		print self.all_concepts
		print inp
		querry_dict = {self.model.input_sequence : inp[0], self.model.input_sequence_lengths: inp[1], self.model.input_comp:self.all_concepts}
		#querry_dict = {self.model.input_sequence : inp[0], self.model.input_sequence_lengths: inp[1], self.model.input_comp:self.all_concepts}
		res_querry = self.sess.run(self.model.querry_distance, feed_dict = querry_dict)
		res_final = self.sess.run(self.model.final_distance, feed_dict = querry_dict)
		#board_str, res = self.sess.run([self.merged_summaries, self.querry_distance], feed_dict = {self.model.input_sequence : inp[0], self.model.input_sequence_lengths: inp[1], self.model.input_comp: self.all_concepts})
#		board_str = self.sess.run(self.merged_summaries, feed_dict = {self.model.input_sequence : inp[0], self.model.input_sequence_lengths: inp[1]})
		#self.board_writer.add_summary(board_str, count)
		indecies = np.argsort(res_final[0,:])
		
		for i in range(5):
			print "------------------------------------------"

		num_printed = 0
		for i in indecies:
			print self.rd.concepts[i], self.rd.names[self.rd.concepts[i]], res_final[0,i], res_querry[0,i]
			num_printed += 1
			if res_final[0,i] >= 1.0: # or num_printed>10:
				break


		
		
	def __init__(self):

		oboFile = open("hp.obo")
		vectorFile = open("vectors.txt")
		#vectorFile = open("train_data_gen/test_vectors.txt")

		self.rd = reader.Reader(oboFile, vectorFile)
		self.newConfig = train_oe.newConfig
		self.newConfig.vocab_size = self.rd.word2vec.shape[0]
		self.newConfig.word_embed_size = self.rd.word2vec.shape[1]
		self.newConfig.max_sequence_length = self.rd.max_length
		self.newConfig.hpo_size = len(self.rd.concept2id)

		self.model = NCRModel(self.newConfig)
		#tf.image_summary("rep",tf.expand_dims(tf.expand_dims(self.model.final_distance, 1),3), 3)
		#self.merged_summaries = tf.merge_all_summaries()

		self.querry_distance = self.model.querry_distance
		self.all_concepts = np.array(list(self.rd.concept_id_list))

		#init_op=tf.initialize_all_variables()
		saver = tf.train.Saver()
		self.sess = tf.Session()

		#sess.run(init_op)
		saver.restore(self.sess, 'checkpoints/training.ckpt')
		#self.board_writer = tf.train.SummaryWriter("board/",self.sess.graph)


	
def main():
	ant = NeuralAnnotator()
	
	count = 0
	print ">>"
	while True:
		line = sys.stdin.readline().strip()
		ant.get_hp_id(line, count)
		count += 1
		
		if line == '':
			break


if __name__ == '__main__':
	main()

