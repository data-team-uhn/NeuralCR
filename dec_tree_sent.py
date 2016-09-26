import numpy as np
from sklearn import linear_model
import accuracy
import reader
import tensorflow as tf
import argparse
import sys
import requests
import json
import cPickle as pickle
import random
from os import listdir
from blist import sortedlist
from sent_level import TextAnnotator

class Ensemble:
	def prepare_pairs(self, phrases):
		ncr_pred = self.textAnt.process_phrase(phrases, 10)
		pht_pred = self.textAnt.phenotips(phrases, 10)
		pair_list = []
		for i in range(len(phrases)):
			ncr_dic = dict(ncr_pred[i])
			pht_dic = dict(pht_pred[i])
			ncr_inf = 4
			pht_inf = 0
			hpo_terms = set(ncr_dic.keys() + pht_dic.keys())
			tmp_pair_list = []
			for x in hpo_terms:
				if x not in ncr_dic:
					ncr_dic[x] = ncr_inf
				if x not in pht_dic:
					pht_dic[x] = pht_inf

				tmp = (ncr_dic[x], pht_dic[x], x)
				tmp_pair_list.append(tmp)
			pair_list.append(tmp_pair_list)
		return pair_list


	def create_entries_from_file(self, sampleFile):
		samples = accuracy.prepare_phrase_samples(self.textAnt.ant.rd, sampleFile)
		entries = [(samples[s][0],s) for s in samples]
		return entries

		'''
			q, _, p = line.split("\t")[0].replace("[", "").replace("]","").strip().split(":")
			q, p = int(q), int(p)
			entry = [x.strip() for x in line.split("\t")[1].split("|")]
			entry[0] = self.textAnt.ant.rd.real_id[entry[0].replace("_",":")]
			if entry[0] in self.textAnt.ant.rd.concepts and entry[1] not in [x[1] for x in entries]: 
				entries.append(entry)
		return entries
		'''



	def __init__(self, repdir = None):
		self.textAnt = TextAnnotator(repdir, "data/")

	def load_logreg(self, logreg_file_adr):
		self.logreg = pickle.load(open(logreg_file_adr,"rb"))

	def train(self, samples):
		data = []
		tmp_data = self.prepare_pairs([entry[1] for entry in samples])
		for i,entry in enumerate(samples):
			data += [(x[0], x[1], x[2]==entry[0]) for x in tmp_data[i]]

		X = np.array([[x[0], x[1]] for x in data])
		Y = np.array([x[2] for x in data])
		self.logreg = linear_model.LogisticRegression(C=1e5)
		self.logreg.fit(X, Y)
		pickle.dump(self.logreg, open("ensemble.p","wb"))


	def predict(self, phrases, top_size=1):
		candidates_all_phrases = self.prepare_pairs(phrases)
		all_scored_candidates = []
		for candidates in candidates_all_phrases:
			scored_candidates = []
			candidates_array = np.array([[c[0],c[1]] for c in candidates])
			scores = self.logreg.predict_proba(candidates_array)
			for i,c in enumerate(candidates):
				scored_candidates.append((c[2], scores[i,1]))
			scored_candidates = sorted(scored_candidates, key= lambda c : -c[1])
			all_scored_candidates.append(scored_candidates[:top_size])
		return all_scored_candidates


def find_phrase_accuracy(func, samples, top_size=None, verbose=False):
	header = 0
	batch_size = 64

	hit=0
	nohit=0
	bad_samples = []

	while header < len(samples):
		batch = samples[header:min(header+batch_size, len(samples))]
		header += batch_size
		results = func([x[1] for x in batch], top_size)
		for i,s in enumerate(batch):
			if s[0] in [x[0] for x in results[i]]:
				hit += 1
			else:
				nohit += 1
				bad_samples.append(s[1])
				if verbose:
					print "---------------------"
					print s, batch[s], ant.rd.names[batch[s][0]]
					print ""
					for res in results[i]:
						print res[0], ant.rd.names[res[0]], res[1]
	total = hit + nohit
	return hit, total



def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--repdir', help="The location where the checkpoints are stored, default is \'checkpoints/\'", default="checkpoints/")
	parser.add_argument('--ans_dir')
	parser.add_argument('--input')
	parser.add_argument('--output_dir')
	parser.add_argument('--threshold', type=float, default=1.0)
	parser.add_argument('--filter_overlap', action='store_true', default=False)
	args = parser.parse_args()


	'''
	random.seed(0)
	random.shuffle(samples)
	training_size = 100
	training = samples[:training_size]
	test = samples[training_size:]
	'''
	
#	ensemble = Ensemble(None, args.repdir, training, args.ans_dir)
#	pickle.dump(ensemble.logreg, open("ensemble.p","wb"))
	ensemble = Ensemble(args.repdir)
	samples = ensemble.create_entries_from_file(open("data/labeled_data"))

	random.seed(0)
	random.shuffle(samples)
	training_size = 500
	training = samples[:training_size]
	test = samples[training_size:]

	ensemble.train(training)
	
	
	print ensemble.predict(["eye disease", "kidney cancer"])


	top_size = 10
	hit, total = find_phrase_accuracy(ensemble.predict, test, top_size)
	print hit, total, float(hit)/total
	hit, total = find_phrase_accuracy(ensemble.textAnt.phenotips, test, top_size)
	print hit, total, float(hit)/total
	hit, total = find_phrase_accuracy(ensemble.textAnt.process_phrase, test, top_size)
	print hit, total, float(hit)/total
#	 	print ensemble.predict([x[1] for x in entries])

	exit()
	print test
	'''	
	print "--"
	print logreg.predict_proba(X)
	'''	



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
