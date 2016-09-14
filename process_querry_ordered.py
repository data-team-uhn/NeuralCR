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
import argparse

def prepare_samples(rd, samplesFile):
	samples = {}
	ct = 0
	for line in samplesFile:
		tokens = line.strip().split("\t")
		real_hp_id = rd.real_id[tokens[1].strip().replace("_",":")]
		if real_hp_id not in rd.concepts:
			continue
		samples[tokens[0].strip()] = [real_hp_id]
		ct += 1
		if ct == 500000:
			break
	return samples

class NeuralAnnotator:

	def get_hp_id(self, querry, count=5):
		inp = self.rd.create_test_sample(querry)
		querry_dict = {self.model.input_sequence : inp['seq'], self.model.input_stemmed_sequence : inp['stem_seq'], self.model.input_sequence_lengths: inp['seq_len']}
		res_querry = self.sess.run(self.querry_distances, feed_dict = querry_dict)
		results=[]
		for s in range(len(querry)):
			indecies_querry = np.argsort(res_querry[s,:])
			tmp_res = []
			num_printed = 0
			for i in indecies_querry:
				num_printed += 1
				tmp_res.append((self.rd.concepts[i],res_querry[s,i]))
				if num_printed>=count:
					break
			results.append(tmp_res)
		return results
	def get_hp_id_comp_phrase(self, querry, count=5):
		inp = self.rd.create_test_sample(querry)
		querry_dict = {self.model.input_sequence : inp['seq'], self.model.input_stemmed_sequence : inp['stem_seq'], self.model.input_sequence_lengths: inp['seq_len']}
		res_querry = self.sess.run(self.querry_distances_phrases, feed_dict = querry_dict)
		results=[]
		for s in range(len(querry)):
			indecies_querry = np.argsort(res_querry[s,:])
			tmp_res = []
			for i in indecies_querry:
				res_item = (self.rd.concepts[self.rd.name2conceptid.values()[i]],res_querry[s,i])
				if res_item[0] not in [x[0] for x in tmp_res]:
					tmp_res.append(res_item)
				if len(tmp_res)>=count:
					break
			results.append(tmp_res)
		return results

	def find_accuracy(self, samples, top_size):
		header = 0
		batch_size = 64

		hit=[0]*top_size
		nohit=0

		while header < len(samples):
			batch = {x:samples[x] for x in samples.keys()[header:min(header+batch_size, len(samples))]}
			header += batch_size
			results = self.get_hp_id_comp_phrase(batch.keys(),top_size)
			#results = self.get_hp_id(batch.keys(),top_size)
			for i,s in enumerate(batch):
				hashit = False
				for attempt,res in enumerate(results[i]):
					if res[0] in batch[s]:
						hit[attempt] += 1
						hashit = True
						break
				if not hashit:
					nohit += 1
					if self.verbose:
						print "---------------------"
						print s, batch[s], self.rd.names[batch[s][0]]
						print ""
						for res in results[i]:
							print res[0], self.rd.names[res[0]], res[1]
		total = sum(hit) + nohit
#		print 1.0 - float(nohit)/total
#		print hit, nohit
		return total - nohit, total
		
	def __init__(self, model, rd ,sess):
		self.model=model
		self.rd = rd
		self.sess = sess
		self.querry_distances = self.model.get_querry_dis()
		self.verbose = False

		print "hello1"
		inp = self.rd.create_test_sample(self.rd.name2conceptid.keys())
		print "hello2"
		querry_dict = {self.model.input_sequence : inp['seq'], self.model.input_stemmed_sequence : inp['stem_seq'], self.model.input_sequence_lengths: inp['seq_len']}
		res_querry = self.sess.run(self.model.gru_state, feed_dict = querry_dict)
		print res_querry
		print res_querry.shape
		print "hello3"

		ref_vecs = tf.Variable(res_querry, False)
		sess.run(tf.assign(ref_vecs, res_querry))

		self.querry_distances_phrases = self.model.euclid_dis_cartesian(ref_vecs, self.model.gru_state)


	
def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--repdir', help="The location where the checkpoints are stored, default is \'checkpoints/\'", default="checkpoints/")
	parser.add_argument('--verbose', help="Print incorrect predictions", action='store_true', default=False)
	args = parser.parse_args()

	oboFile = open("hp.obo")
	vectorFile = open("vectors.txt")
	samplesFile = open("labeled_data")

	stemmedVectorFile = open("stemmed_vectors.txt")
	rd = reader.Reader(oboFile, vectorFile, stemmedVectorFile)
	'''
	print rd.concepts[9791]
	print len(rd.name2conceptid)
	exit()
	'''
	#rd = reader.Reader(oboFile, vectorFile)

	newConfig = train_oe.newConfig
	newConfig.vocab_size = rd.word2vec.shape[0]
	newConfig.stemmed_vocab_size = rd.stemmed_word2vec.shape[0]
	newConfig.word_embed_size = rd.word2vec.shape[1]
	newConfig.max_sequence_length = rd.max_length
	newConfig.hpo_size = len(rd.concept2id)
	newConfig.last_state = True

	model = NCRModel(newConfig)

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, args.repdir + '/training.ckpt')

	ant = NeuralAnnotator(model, rd, sess)
	ant.verbose = args.verbose
	samples = prepare_samples(rd, samplesFile)
	training_samples = {}
	for hpid in rd.names:
		for s in rd.names[hpid]:
			training_samples[s]=[hpid]

	top_size = 5

	hit, total = ant.find_accuracy(samples, top_size)
	print hit, total, float(hit)/total
	exit()
	'''
	hit, total = ant.find_accuracy(training_samples, top_size)
	print hit, total, float(hit)/total
	exit()
	'''
	
	print ">>"
	while True:
		line = sys.stdin.readline().strip()
		res = ant.get_hp_id_comp_phrase([line], top_size)
		for r in res:
			for rid in r:
				print rid[0], rd.names[rid[0]], rid[1]

		
		if line == '':
			break


if __name__ == '__main__':
	main()

