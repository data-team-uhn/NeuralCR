import os
import argparse
import reader
import re
import cPickle as pickle

def get_labeled_sentences(ans_adr):
	oboFile = open("data/hp.obo")
	vectorFile = open("data/vectors.txt")
	rd = reader.Reader(oboFile, vectorFile)

	text_adr = '../data/text/'
	total_results = {}
	for f in os.listdir(text_adr):
		text = open(text_adr+f).read()
		sents = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

		anotations = [x for x in open(ans_adr+f).read().strip().split("\n") if len(x)>0]
		anotations = [x.split("\t") for x in anotations]
	#	anotations = [[map(int,x[0].replace("[","").replace("]","").split("::")), x[1].split("|")[0].strip().replace("_",":")] for x in anotations]
		anotations = [[map(int,x[0].replace("[","").replace("]","").split("::")), rd.real_id[x[1].split("|")[0].strip().replace("_",":")]] for x in anotations]
#		print anotations

		final_results = {}
		for sent in sents:
			start = text.find(sent)
			end = start + len(sent)
			hp_list = []
			for ant in anotations:
				if ant[0][0]>=start-1 and ant[0][1] < end+1:
					if ant[1] in rd.concepts:
						hp_list.append(ant[1])
			final_results[sent]=set(hp_list)
		
		total_used_ants = sum([len(final_results[x]) for x in final_results])
		'''
		if total_used_ants != len(anotations):
			print "============================"
			print f
			print total_used_ants, len(anotations)
			used_ants = [x for pair in final_results.items() for x in pair[1]]
			print [x[1] for x in anotations if x[1] not in used_ants]
			print "Error!!"
		'''
		total_results.update(final_results)
		'''
		for res in final_results:
			print res
			print final_results[res]
			print "--"
		'''
	return total_results

def main():
	ans_adr = '../data/stand-off/'
	#ans_adr = '../../biolark_full_results/'
	results = get_labeled_sentences(ans_adr)
	#pickle.dump(results, open('biolark_results.p', "wb"))
	pickle.dump(results, open('labeled_sentences.p', "wb"))
	
if __name__ == "__main__":
	main()
