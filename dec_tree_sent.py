from process_querry_ordered import NeuralAnnotator
import numpy as np
from sklearn import linear_model
from ordered_embeding import NCRModel
import reader
import train_oe
import tensorflow as tf
import argparse
import sys
import requests
import json
import cPickle as pickle
import random
from os import listdir
from blist import sortedlist

class TextAnnotator:

	def phenotips(self, phrases):
		results = []
		for phrase in phrases:
			resp = requests.get('https://phenotips.org/get/PhenoTips/SolrService?vocabulary=hpo&q='+phrase.replace(" ","+")).json()
			ans = [(x[u'id'],x[u'score']) for x in resp['rows'][:10]]
			results.append(ans)
		return results
	def ncr(self, phrases):
		return self.ant.get_hp_id_comp_phrase(phrases, count=10)

	def process_phrase(self, phrases):
		#		ans_phenotips = self.phenotips(phrases)
		ans_ncr = self.ncr(phrases)
		return ans_ncr
#		return ans_phenotips

	def process_sent(self, sent, threshold, filter_overlap=False):
		tokens = sent.strip().split(" ")
		ret = {}
		candidates = []
		for i,w in enumerate(tokens):
			phrase = ""
			for r in range(5):
				if i+r >= len(tokens):
					break
				phrase += " " + tokens[i+r]
				candidates.append(phrase.strip())
		return candidates


	def process_text(self, text, threshold, filter_overlap=False):
		sents = text.split(".")
		ans = []
		total_chars=0
		final_results = []
		for sent in sents:
			results = self.process_sent(sent, threshold, filter_overlap)
			final_results += results
		return final_results

	def __init__(self, repdir):
		oboFile = open("hp.obo")
		vectorFile = open("vectors.txt")
		samplesFile = open("labeled_data")
		stemmedVectorFile = open("stemmed_vectors.txt")

		self.rd = reader.Reader(oboFile, vectorFile, stemmedVectorFile)

		newConfig = train_oe.newConfig
		newConfig.vocab_size = self.rd.word2vec.shape[0]
		newConfig.stemmed_vocab_size = self.rd.stemmed_word2vec.shape[0]
		newConfig.word_embed_size = self.rd.word2vec.shape[1]
		newConfig.max_sequence_length = self.rd.max_length
		newConfig.hpo_size = len(self.rd.concept2id)
		newConfig.last_state = True

		model = NCRModel(newConfig)

		sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(sess, repdir + '/training.ckpt')

		self.ant = NeuralAnnotator(model, self.rd, sess)


def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--repdir', help="The location where the checkpoints are stored, default is \'checkpoints/\'", default="checkpoints/")
	parser.add_argument('--text_dir')
	parser.add_argument('--ans_dir')
	parser.add_argument('--input')
	parser.add_argument('--output_dir')
	parser.add_argument('--threshold', type=float, default=1.0)
	parser.add_argument('--filter_overlap', action='store_true', default=False)
	args = parser.parse_args()

	samples = listdir(args.text_dir)
	random.shuffle(samples)
	training_size = 100
	training = samples[:training_size]
	test = samples[training_size:]

	textAnt = TextAnnotator(args.repdir)

	data = []

	total = [0.0, 0.0]
	count = [0, 0]

	for sample in training:
		text = open(args.text_dir+"/"+sample).read()
		reduced_text = text
		for line in open(args.ans_dir+"/"+sample):
			q, _, p = line.split("\t")[0].replace("[", "").replace("]","").strip().split(":")
			q, p = int(q), int(p)
			reduced_text = reduced_text[:q] + " " + reduced_text[p:]
			entry = [x.strip() for x in line.split("\t")[1].split("|")]
			entry[0] = textAnt.rd.real_id[entry[0].replace("_",":")]
			ncr_pred = textAnt.ncr([entry[1]])[0]
			pht_pred = textAnt.phenotips([entry[1]])[0]
			ncr_dic = dict(ncr_pred)
			pht_dic = dict(pht_pred)
			triplets = []
			ncr_inf = 4
			pht_inf = 0
			hpo_terms = set(ncr_dic.keys() + pht_dic.keys())
			for x in hpo_terms:
				if x not in ncr_dic:
					ncr_dic[x] = ncr_inf
				if x not in pht_dic:
					pht_dic[x] = pht_inf

				tmp = (ncr_dic[x], pht_dic[x], entry[0]==x)
				data.append(tmp)
		break

	print data
	X = np.array([[x[0], x[1]] for x in data])
	Y = np.array([x[2] for x in data])
	print X
	print Y
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(X, Y)



	exit()
	for x in data:
		print x

#		print reduced_text



	exit()




	if args.input_dir is not None:
		for f in listdir(args.input_dir):
			text = open(args.input_dir+"/"+f).read()
			#print "------------------\n" + text + "\n\n\n\n"
			results = textAnt.process_text(text, args.threshold, args.filter_overlap)
			outf = open(args.output_dir+"/"+f, "w")
			for res in results:
				outf.write(res[2].replace(":","_")+"\n")
				#			print "["+str(res[0])+"::"+str(res[1])+"]\t" , res[2], "|", text[res[0]:res[1]]
		return

	if args.input is not None:
		text = open(args.input).read()

	while True:
		if args.input is None:
			text = sys.stdin.readline()
		if text == "":
			break

		results = textAnt.process_text(text, args.threshold, args.filter_overlap)
		for res in results:
			print "["+str(res[0])+"::"+str(res[1])+"]\t" , res[2], "|", text[res[0]:res[1]], "\t", res[3], "\t", textAnt.rd.names[res[2]]
		if args.input is not None:
			break


#	text = "We report on seven children with Angelman syndrome presenting with psychomotor retardation during the 1st year of life. Seizures developed in six patients, and computed tomography (CT) scanning showed diffuse atrophy of the brain in five patients. We conclude that diagnosis is difficult in the first years of life. A review of the literature is given."


	#print [x[0] for x in ant.get_hp_id(["big head", "small head"])]


if __name__ == '__main__':
	main()
