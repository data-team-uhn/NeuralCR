import tensorflow as tf
import cPickle as pickle
import numpy as np
from triplet_reader import DataReader
#from concept2vec import concept_vector_model
import ncr_cnn_model
import sys

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
		if not check_phrase(querry, self.wordList, self.modelConfig.max_num_of_words):
			return None
		querry_embedding = np.reshape ( embed_phrase(querry, self.wordList, self.modelConfig.max_num_of_words), (1, self.modelConfig.max_num_of_words * self.modelConfig.word_size))
		querry_rep = self.sess.run(self.m.rep, feed_dict={self.m.input_vectors : querry_embedding})
		cosine_sim = np.matmul(querry_rep, np.transpose(self.hpo_reps))
		querry_hpo_embedding_index = np.argmax(cosine_sim, axis=1)
		querry_hpo = self.embedded_concepts_indecies[querry_hpo_embedding_index]
		querry_name = self.concepts[querry_hpo]['names']
		
		return querry_hpo
		
	def __init__(self):
		self.modelConfig = ncr_cnn_model.bigConfig()
		with tf.variable_scope("models"):
			self.m = ncr_cnn_model.NCRModel(self.modelConfig)

		self.concepts = pickle.load(open('hpo.pickle','rb'))
		self.wordList = pickle.load(open('word-vectors.pickle','rb'))

		embedded_concepts = []
		self.embedded_concepts_indecies = []
		for hp in self.concepts:
			for phrase in self.concepts[hp]['names']:
				if check_phrase(phrase, self.wordList, self.modelConfig.max_num_of_words):
					embedded_concepts.append(np.reshape(embed_phrase( phrase, self.wordList, self.modelConfig.max_num_of_words), (1, self.modelConfig.max_num_of_words * self.modelConfig.word_size)))
					self.embedded_concepts_indecies.append(hp)
		
		hpo_embedings = np.concatenate(embedded_concepts, axis = 0)

		#init_op=tf.initialize_all_variables()
		saver = tf.train.Saver()
		self.sess = tf.Session()
		#sess.run(init_op)
		saver.restore(self.sess, 'checkpoints/training.ckpt')

		self.hpo_reps = self.sess.run(self.m.rep, feed_dict={self.m.input_vectors : hpo_embedings})
		print self.hpo_reps.shape

	
def main():
	ant = NeuralAnnotator()
	while True:
		line = sys.stdin.readline().strip()
		hp_id = ant.get_hp_id(line)
		if hp_id == None:
			print "bad input"
		else:
			print hp_id, ant.concepts[hp_id]['names']
		
		if line == '':
			break


if __name__ == '__main__':
	main()

