from process_querry_ordered import NeuralAnnotator
from ordered_embeding import NCRModel
import reader
import train_oe
import tensorflow as tf
import argparse
import sys
import requests
import json
import cPickle as pickle
from os import listdir
from blist import sortedlist

class TextAnnotator:

	def phenotips(self, phrases):
		results = []
		for phrase in phrases:
			resp = requests.get('https://phenotips.org/get/PhenoTips/SolrService?vocabulary=hpo&q='+phrase.replace(" ","+")).json()
			ans = resp['rows'][:1][0]
			results.append([(ans[u'id'],1.0/ans[u'score'])])
		return results

	def process_phrase(self, phrases):
		#		ans_phenotips = self.phenotips(phrases)
		ans_ncr = self.ant.get_hp_id_comp_phrase(phrases, count=1)
		return ans_ncr
#		return ans_phenotips

	def process_sent(self, sent, threshold, filter_overlap=False):
		tokens = sent.strip().split(" ")
		ret = {}
		for i,w in enumerate(tokens):
			phrase = ""
			candidates = []
			for r in range(5):
				if i+r >= len(tokens):
					break
				phrase += " " + tokens[i+r]
				candidates.append(phrase.strip())
			hp_ids = self.process_phrase(candidates)
			#print hp_ids
			for i in range(len(hp_ids)):
				if hp_ids[i][0][1] < threshold:
					if (hp_ids[i][0][0] not in ret) or (hp_ids[i][0][1]<ret[hp_ids[i][0][0]][0]):
						ret[hp_ids[i][0][0]] = (hp_ids[i][0][1], candidates[i])
		results = []
		for hp_id in ret:
			results.append([sent.index(ret[hp_id][1]), sent.index(ret[hp_id][1])+len(ret[hp_id][1]), hp_id, ret[hp_id][0]])

		results = sorted(results, key=lambda x : (x[3], x[0]-x[1]))
#		print results

		if filter_overlap:
			filtered_results = sortedlist([], key = lambda x : x[0])
			for res in results:
				match = filtered_results.bisect(res)
				'''
				print "--"
				print res
				print filtered_results
				print match
				'''
				if match==0 or filtered_results[match-1][1]<res[1]:
					#print "added!"
					filtered_results.add(res)
			return list(filtered_results)
		else:
			return results

	def process_text(self, text, threshold, filter_overlap=False):
		sents = text.split(".")
		ans = []
		total_chars=0
		final_results = []
		for sent in sents:
			results = self.process_sent(sent, threshold, filter_overlap)
			for i in range(len(results)):
				results[i][0] += total_chars
				results[i][1] += total_chars
			final_results += results
			total_chars += len(sent)+1
		final_results = sorted(final_results, key=lambda x : x[0])
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
	parser.add_argument('--input_dir')
	parser.add_argument('--input')
	parser.add_argument('--output_dir')
	parser.add_argument('--threshold', type=float, default=1.0)
	parser.add_argument('--filter_overlap', action='store_true', default=False)
	args = parser.parse_args()

	textAnt = TextAnnotator(args.repdir)
	for f in listdir(args.input_dir):
		text = open(args.input_dir+"/"+f).read()
		#print "------------------\n" + text + "\n\n\n\n"
		results = textAnt.process_text(text, args.threshold, args.filter_overlap)
		outf = open(args.output_dir+"/"+f, "w")
		for res in results:
			outf.write(res[2].replace(":","_")+"\n")
			#			print "["+str(res[0])+"::"+str(res[1])+"]\t" , res[2], "|", text[res[0]:res[1]]
	exit()

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
