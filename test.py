import fasttext_reader as reader
#import reader
import phrase_model
import accuracy
import phrase_annotator
import phraseConfig
import tensorflow as tf
import sys
import requests
import json

class PhenotipsWrapper:
	def get_hp_id(self, phrases, count=1):
		results = []
		for phrase in phrases:
			resp = requests.get('https://phenotips.org/get/PhenoTips/SolrService?vocabulary=hpo&q='+phrase.replace(" ","+"), verify=False).json()
			ans = [(self.rd.real_id[str(x[u'id'])],x[u'score']) if str(x[u'id']) in self.rd.real_id else (x[u'id'],x[u'score']) for x in resp['rows'][:count]]
			results.append(ans)
		return results
	def __init__(self, rd):
		self.rd = rd

def test_accuarcy_phrase():
	samplesFile = open("data/labeled_data")

	oboFile = open("data/hp.obo")
	vectorFile = open("data/vectors.txt")
	rd = reader.Reader(oboFile)

#	ant = phrase_annotator.create_annotator("checkpoints/", "data/", False, False)
	ant = phrase_annotator.create_annotator("checkpoints", "data/", True, False)
	#ant = phrase_annotator.create_annotator("/ais/gobi4/arbabi/codes/NeuralCR/checkpoints", "data/", True, False)
	#ant = phrase_annotator.create_annotator("/ais/gobi4/arbabi/codes/NeuralCR/checkpoints", "data/", True, False)
	
	#ant = phrase_annotator.create_annotator("checkpoints_backup/", "data/", True, False)
#	ant = PhenotipsWrapper(rd)

	samples = accuracy.prepare_phrase_samples(rd, samplesFile)
	cor, tot = accuracy.find_phrase_accuracy(ant, samples, 5, True)
	print cor, tot
	print float(cor)/tot

def main():
	test_accuarcy_phrase()


if __name__ == "__main__":
	main()
