import tensorflow as tf
import numpy as np
import fasttext_reader as reader
#import reader
import phraseConfig
import phrase_model
import sys

class NeuralPhraseAnnotator:
	def __get_top_concepts(self, indecies_querry, res_querry, count):
		tmp_res = []
		for i in indecies_querry:
			tmp_res.append((self.rd.concepts[i],res_querry[i]))
			if len(tmp_res)>=count:
				break
		return tmp_res

	def get_hp_id(self, querry, count=1):
		inp = self.rd.create_test_sample(querry)
		querry_dict = {self.model.input_sequence : inp['seq'], self.model.input_sequence_lengths: inp['seq_len']}
		#res_querry = -self.sess.run(self.model.layer4, feed_dict = querry_dict)
		#res_querry = -self.sess.run(self.model.score_layer, feed_dict = querry_dict)
		res_querry = 1.0-self.sess.run(self.model.pred, feed_dict = querry_dict)

		results=[]
		for s in range(len(querry)):
			indecies_querry = np.argsort(res_querry[s,:])
			results.append(self.__get_top_concepts(indecies_querry, res_querry[s,:], count))

		return results

	def __init__(self, model, rd ,sess, compWithPhrases = False):
		self.model=model
		self.rd = rd
		self.sess = sess
		return
		self.compWithPhrases = compWithPhrases

		if self.compWithPhrases:
			inp = self.rd.create_test_sample(self.rd.name2conceptid.keys())
			querry_dict = {self.model.input_sequence : inp['seq'], self.model.input_sequence_lengths: inp['seq_len']}
			res_querry = self.sess.run(self.model.gru_state, feed_dict = querry_dict)
			ref_vecs = tf.Variable(res_querry, False)
			sess.run(tf.assign(ref_vecs, res_querry))
			self.querry_distances = self.model.euclid_dis_cartesian(ref_vecs, self.model.gru_state)
		else:
			self.querry_distances = self.model.euclid_dis_cartesian(self.model.get_HPO_embedding(), self.model.gru_state)\
					+ self.model.order_dis_cartesian(self.model.get_HPO_embedding(), self.model.gru_state)
#					+ self.model.order_dis_cartesian(self.model.get_HPO_embedding(), self.model.gru_state)
#			self.querry_distances = 0\
		#					+ self.model.order_dis_cartesian(self.model.get_HPO_embedding(), self.model.gru_state)

def create_annotator(repdir, datadir=None, compWithPhrases = False, addNull=False):
	if datadir is None:
		datadir = repdir
	oboFile = open(datadir+"/hp.obo")
	vectorFile = open(datadir+"/vectors.txt")

	rd = reader.Reader(oboFile) #, vectorFile, addNull)
	config = phraseConfig.Config
	config.update_with_reader(rd)

	model = phrase_model.NCRModel(config, ancs_sparse = rd.sparse_ancestrs)

	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	saver = tf.train.Saver()
	saver.restore(sess, repdir + '/training.ckpt')

	return NeuralPhraseAnnotator(model, rd, sess, compWithPhrases)

def main():
	#ant = create_annotator("checkpoints/", "data/", False, False)
	ant = create_annotator("/ais/gobi4/arbabi/codes/NeuralCR/checkpoints", "data/", True, False)
	#ant = create_annotator("checkpoints_backup/", "data/", True, False)
	while True:
		sys.stdout.write("-----------\nEnter text:\n")
		sys.stdout.flush()
		text = sys.stdin.readline()
		sys.stdout.write("\n")
		matches = ant.get_hp_id([text],15)
		for x in matches[0]:
			sys.stdout.write(x[0]+' '+str(ant.rd.names[x[0]])+' '+str(x[1])+'\n')
		sys.stdout.write("\n")
	
	return
	print ant.get_hp_id(["kindey", "renal"],5)
	print [ant.rd.names[x[0]] for x in  ant.get_hp_id(["kindey", "renal"],5)]
	return
	words =["brain retardation"] #, "kindey", "renal"]
	for i,item in enumerate(ant.get_hp_id(words,20)):
		print "-------"
		print words[i]
		for x in item:
			print x[0], ant.rd.names[x[0]], x[1]
	return
	sample_data = open("cui_hpo_translations.csv").read().split("\n")
	cui = [x.split(",")[0] for x in sample_data if len(x) > 0]
	sample_data = [x.split(",")[1].replace('\"','') for x in sample_data if len(x) > 0]
	results = ant.get_hp_id(sample_data, 5)
	for i,x in enumerate(cui):
		sys.stdout.write(cui[i] + "," + sample_data[i]+ "," + ("%.4f" % results[i][0][1]))
		for j in range(5):
			sys.stdout.write("," + results[i][j][0] +"," + ant.rd.names[results[i][j][0]][0].replace(",","-"))
		sys.stdout.write("\n")


if __name__ == "__main__":
	main()

