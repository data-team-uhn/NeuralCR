import annotator
import tensorflow as tf
import argparse
import sys
import requests
import json
import cPickle as pickle
from os import listdir
from blist import sortedlist
import gpu_access
import time

class TextAnnotator:

	def phenotips(self, phrases, count=1):
		results = []
		for phrase in phrases:
			resp = requests.get('https://phenotips.org/get/PhenoTips/SolrService?vocabulary=hpo&q='+phrase.replace(" ","+")).json()
			ans = [(self.ant.rd.real_id[str(x[u'id'])],x[u'score']) if str(x[u'id']) in self.ant.rd.real_id else (x[u'id'],x[u'score']) for x in resp['rows'][:count]]
			results.append(ans)
		return results

	def process_phrase(self, phrases, count=1):
		with tf.device('/gpu:'+self.board):
			ans_ncr = self.ant.get_hp_id(phrases, count)
		return ans_ncr

	def process_sent(self, sent, threshold, filter_overlap=False):
		tokens = sent.strip().split(" ")
		ret = {}
		for i,w in enumerate(tokens):
			phrase = ""
			candidates = []
			for r in range(7):
				if i+r >= len(tokens):
					break
				phrase += " " + tokens[i+r]
				candidates.append(phrase.strip())
			hp_ids = self.process_phrase(candidates, 1)
			for i in range(len(hp_ids)):
				if hp_ids[i][0][1] < threshold:
					if (hp_ids[i][0][0] not in ret) or (hp_ids[i][0][1]<ret[hp_ids[i][0][0]][0]):
						ret[hp_ids[i][0][0]] = (hp_ids[i][0][1], candidates[i])
		results = []
		for hp_id in ret:
			results.append([sent.index(ret[hp_id][1]), sent.index(ret[hp_id][1])+len(ret[hp_id][1]), hp_id, ret[hp_id][0]])
		
		results = sorted(results, key=lambda x : (x[3], x[0]-x[1]))

		if filter_overlap:
			filtered_results = sortedlist([], key = lambda x : x[0])
			for res in results:
				match = filtered_results.bisect(res)
				if match==0 or filtered_results[match-1][1]<res[1]:
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

	def __init__(self, repdir, datadir=None):
		self.board = gpu_access.get_gpu()
		with tf.device('/gpu:'+self.board):
			self.ant = annotator.create_annotator(repdir, datadir, True)



def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--repdir', help="The location where the checkpoints are stored, default is \'checkpoints/\'", default="checkpoints/")
	parser.add_argument('--input_dir')
	parser.add_argument('--input')
	parser.add_argument('--output_dir')
	parser.add_argument('--threshold', type=float, default=1.0)
	parser.add_argument('--filter_overlap', action='store_true', default=False)
	args = parser.parse_args()

	sys.stderr.write("Initializing NCR...\n")
	textAnt = TextAnnotator(args.repdir, "data/")
	sys.stderr.write("Done.\n")

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
			print "Enter querry:"
			text = sys.stdin.readline()
		if text == "":
			break
		start_time = time.time()
		results = textAnt.process_text(text, args.threshold, args.filter_overlap)
		end_time = time.time()
		for res in results:
			print "["+str(res[0])+"::"+str(res[1])+"]\t" , res[2], "|", text[res[0]:res[1]], "\t", res[3], "\t", textAnt.ant.rd.names[res[2]]
		print "Time elapsed: "+ ("%.2f" % (end_time-start_time)) + "s"
		if args.input is not None:
			break


if __name__ == '__main__':
	main()
