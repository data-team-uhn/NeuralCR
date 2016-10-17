import tensorflow as tf
import numpy as np
import reader
import phraseConfig
import sent_model
import sent_accuracy
import argparse
import gpu_access

class NeuralSentenceAnnotator:
	def __get_top_concepts(self, indecies_querry, res_querry, threshold):
		tmp_res = []
		if self.compWithPhrases:
			for i in indecies_querry:
				if res_querry[i]>threshold:
					break
				res_item = (self.rd.concepts[self.rd.name2conceptid.values()[i]],res_querry[i])
				if res_item[0] not in [x[0] for x in tmp_res]:
					tmp_res.append(res_item)
		else:
			for i in indecies_querry:
				if res_querry[i]>threshold:
					break
				tmp_res.append((self.rd.concepts[i],res_querry[i]))
		return tmp_res

	def get_hp_id(self, querries, threshold):
		inp = self.rd.create_test_sample(querries)
		querry_dict = {self.model.input_sequence : inp['seq'], self.model.input_sequence_lengths: inp['seq_len']}
		res_querry = self.sess.run(self.querry_distances, feed_dict = querry_dict)

		results=[]
		for s in range(len(querries)):
			indecies_querry = np.argsort(res_querry[s,:])
			results.append(self.__get_top_concepts(indecies_querry, res_querry[s,:], threshold))

		
		return results

	def process_single_sent(self, sent, threshold=1.0):
		results = self.get_hp_id([sent], threshold)
		return results

	def process_text(self, text, threshold=1.0):
		sents = text.split(".")
		results = self.get_hp_id(sents, threshold)
#		print results
		return results
		'''
		ans = []
		total_chars=0
		final_results = []
		for sent in sents:
			for i in range(len(results)):
				results[i][0] += total_chars
				results[i][1] += total_chars
			final_results += results
			total_chars += len(sent)+1
		final_results = sorted(final_results, key=lambda x : x[0])
		return final_results
		'''


	def __init__(self, model, rd ,sess, compWithPhrases = False):
		self.model=model
		self.rd = rd
		self.sess = sess
		self.compWithPhrases = compWithPhrases

		if self.compWithPhrases:
			inp = self.rd.create_test_sample(self.rd.name2conceptid.keys())
			querry_dict = {self.model.input_sequence : inp['seq'], self.model.input_sequence_lengths: inp['seq_len']}
			res_querry = self.sess.run(self.model.gru_state, feed_dict = querry_dict)
			ref_vecs = tf.Variable(res_querry, False)
			sess.run(tf.assign(ref_vecs, res_querry))
			self.querry_distances = self.model.euclid_dis_cartesian(ref_vecs, self.model.gru_state)
		else:
			self.querry_distances = self.model.rnn_minpool_cartesian(self.model.get_HPO_embedding())
			#self.querry_distances = self.model.euclid_dis_cartesian(self.model.get_HPO_embedding(), self.model.gru_state)

def create_annotator(repdir, datadir=None, compWithPhrases = False, addNull=False):
	if datadir is None:
		datadir = repdir
	oboFile = open(datadir+"/hp.obo")
	vectorFile = open(datadir+"/vectors.txt")

	rd = reader.Reader(oboFile, vectorFile, addNull)
	config = phraseConfig.Config
	config.update_with_reader(rd)

	model = sent_model.NCRModel(config)

	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	saver = tf.train.Saver()
	saver.restore(sess, repdir + '/sentence_training.ckpt')

	return NeuralSentenceAnnotator(model, rd, sess, compWithPhrases)

class Sent_ant_wrapper:
	def process_text(self, text, threshold = 1.0):
		with tf.device('/gpu:'+self.board):
			return self.ant.process_text(text, threshold)

	def __init__(self, repdir, addNull=False):
		self.board = gpu_access.get_gpu()
		with tf.device('/gpu:'+self.board):
			self.ant = create_annotator(repdir, "data/", addNull=addNull)		



def main():
	parser = argparse.ArgumentParser(description='Hello!')
#	parser.add_argument('--repdir', help="The location where the checkpoints are stored, default is \'checkpoints/\'", default="sent_checkpoints_backup/")
	parser.add_argument('--repdir', help="The location where the checkpoints are stored, default is \'checkpoints/\'", default="sent_checkpoints/")
	args = parser.parse_args()
	
	board = gpu_access.get_gpu()
	with tf.device('/gpu:'+board):
		ant = create_annotator(args.repdir, "data/", addNull=True)
#		sent_accuracy.find_sent_accuracy(lambda text: [x[0] for sent_res in ant.process_single_sent(text, 1.0) for x in sent_res], "labeled_sentences.p", ant.rd)
		sent_accuracy.find_sent_accuracy(lambda text: [x[0] for sent_res in ant.process_text(text, 1.0) for x in sent_res], "labeled_sentences.p", ant.rd)
		#sent_accuracy.find_sent_accuracy(lambda text: [x[0] for sent_res in ant.process_text(text, 0.8) for x in sent_res], "labeled_sentences.p")


if __name__ == '__main__':
	main()

