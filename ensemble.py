import process_querry_ordered
import json
import requests
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

def get_res_phenotips(querry, top_size):
	#resp = requests.get('https://phenotips.org/get/PhenoTips/SolrService?vocabulary=hpo&q='+querry.replace(" ","+")).json()
	resp = requests.get('http://192.75.158.11/get/PhenoTips/SolrService?vocabulary=hpo&q='+querry.replace(" ","+")).json()
	return [s[u'id'] for s in resp['rows'][:top_size]]

def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--repdir', help="The location where the checkpoints are stored, default is \'checkpoints/\'", default="checkpoints/")
	parser.add_argument('--verbose', help="Print incorrect predictions", action='store_true', default=False)
	args = parser.parse_args()

	oboFile = open("hp.obo")
	vectorFile = open("vectors.txt")
	samplesFile = open("labeled_data")

	rd = reader.Reader(oboFile, vectorFile)

	newConfig = train_oe.newConfig
	newConfig.vocab_size = rd.word2vec.shape[0]
	newConfig.word_embed_size = rd.word2vec.shape[1]
	newConfig.max_sequence_length = rd.max_length
	newConfig.hpo_size = len(rd.concept2id)
	newConfig.last_state = True

	model = NCRModel(newConfig)

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, args.repdir + '/training.ckpt')

	ant = process_querry_ordered.NeuralAnnotator(model, rd, sess)
	ant.verbose = args.verbose
	samples = process_querry_ordered.prepare_samples(rd, samplesFile)

	phenotips_ans = pickle.load(open('phenotips_ans.p','rb'))

	
	print ">>"
	'''
	while True:
		line = sys.stdin.readline().strip()
	'''
	top_size=10
	hit_ncr = 0
	hit_phenotips = 0
	hit_ensemble = 0
	total = 0
	for s in samples:
		ncr_res = ant.get_hp_id([s], top_size)[0]
		#phenotips_res = get_res_phenotips(s, top_size)
		phenotips_res = phenotips_ans[s][:top_size]
		total += 1
#		print samples[s][0], ncr_res, phenotips_res
		if samples[s][0] in ncr_res:
			hit_ncr += 1
		if samples[s][0] in phenotips_res:
			hit_phenotips += 1
		if samples[s][0] in (phenotips_res+ncr_res):
			hit_ensemble += 1
		'''
		for r in res:
			for rid in r:
				print rid, rd.names[rid]
		'''
	print hit_ncr, total, float(hit_ncr)/total
	print hit_phenotips, total, float(hit_phenotips)/total
	print hit_ensemble, total, float(hit_ensemble)/total

if __name__ == '__main__':
	main()

