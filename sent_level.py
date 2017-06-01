import tensorflow as tf
import argparse
import sys
import requests
import json
import cPickle as pickle
from os import listdir
from blist import sortedlist
import time
import fasttext_reader as reader
from phraseConfig import Config
import phrase_model

class TextAnnotator:
	def phenotips(self, phrases, count=1):
		results = []
		for phrase in phrases:
			resp = requests.get('https://phenotips.org/get/PhenoTips/SolrService?vocabulary=hpo&q='+phrase.replace(" ","+")).json()
			ans = [(self.rd.real_id[str(x[u'id'])],x[u'score']) if str(x[u'id']) in self.ant.rd.real_id else (x[u'id'],x[u'score']) for x in resp['rows'][:count]]
			results.append(ans)
		return results

	def process_phrase(self, phrases, count=1):
		#print phrases
		ans_ncr = self.model.get_hp_id(phrases, count)
       #         for i in range(len(phrases)):
       #             print phrases[i], ans_ncr[i]
		return ans_ncr

	def process_sent(self, sent, threshold, filter_overlap=False):
		tokens = sent.strip().split(" ")
                #tokens = fasttext_reader.tokenize(sent)
		ret = {}
                candidates = []
		for i,w in enumerate(tokens):
			phrase = ""
			for r in range(7):
				if i+r >= len(tokens):
					break
				phrase += " " + tokens[i+r]
                                cand_phrase = phrase.strip(',/;-.').strip()
				if len(cand_phrase) > 0:
					candidates.append(cand_phrase)
                hp_ids = [x[0] for x in self.process_phrase(candidates, 1)]
                for i in range(len(hp_ids)):
                        if hp_ids[i][0]!='HP:0000118' and hp_ids[i][0]!='HP:None' and hp_ids[i][1] > threshold:
                                if (hp_ids[i][0] not in ret) or (hp_ids[i][1]>ret[hp_ids[i][0]][0]):
                                        ret[hp_ids[i][0]] = (hp_ids[i][1], candidates[i])
		results = []
		for hp_id in ret:
			results.append([sent.index(ret[hp_id][1]), sent.index(ret[hp_id][1])+len(ret[hp_id][1]), hp_id, ret[hp_id][0]])
		
		results = sorted(results, key=lambda x : (-x[3], x[0]-x[1]))

		if filter_overlap:
			'''
			filtered_results = []
			for res in results:
				bad = False
				for oth in filtered_results:
					if not (res[1] <= oth[0] or oth[1] <=res[0]):
						bad = True
						break
				if not bad:
					filtered_results.append(res)
			return list(filtered_results)
			'''
			filtered_results = sortedlist([], key = lambda x : x[0])
			for res in results:
				match = filtered_results.bisect(res)
				if match==0 or filtered_results[match-1][1]<res[1]:
					filtered_results.add(res)
			return list(filtered_results)
		else:
			return results

	def process_text_fast_new(self, text, threshold=0.5, filter_overlap=False):
            chunks_large = text.replace("\r"," ").replace("\n"," ").replace("\t", " ").replace(",","|").replace(";","|").replace(".","|").split("|")
            candidates = []
            candidates_info = []
            total_chars=0
            for c,chunk in enumerate(chunks_large):
		tokens = chunk.split(" ")
                chunk_chars = 0
		for i,w in enumerate(tokens):
                    phrase = ""
                    for r in range(7):
                        if i+r >= len(tokens) or len(tokens[i+r])==0:
                            break
                        if r>0:
                            phrase += " " + tokens[i+r]
                        else:
                            phrase = tokens[i+r]
                        #cand_phrase = phrase.strip(',/;-.').strip()
                        cand_phrase = phrase
                        if len(cand_phrase) > 0:
                            candidates.append(cand_phrase)
                            location = total_chars+chunk_chars
                            candidates_info.append((location, location+len(phrase), c))
                    chunk_chars += len(w)+1
                total_chars += len(chunk)+1
            matches = [x[0] for x in self.process_phrase(candidates, 1)]
            filtered = {}
            #print "---->>>>"
            #print matches
            #print " "
            for i in range(len(candidates)):
                if matches[i][0]!='HP:0000118' and matches[i][0]!="HP:None" and matches[i][1]>threshold:
                    if candidates_info[i][2] not in filtered:
                        filtered[candidates_info[i][2]] = []
                    filtered[candidates_info[i][2]].append((candidates_info[i][0], candidates_info[i][1], matches[i][0], matches[i][1]))
            #print filtered
            #print " "
            final = []
            for c in filtered:
                for m in filtered[c]:
                    confilict = False
                    #print " :: ", m
                    for m2 in filtered[c]:
#                        print " :: --", m2,
                        ## m2 and m have some intersection, m2 has better score
                        if m[1]>m2[0] and m[0]<m2[1]:
                            ## m2 fully covers m, another id
                            if m2[0]<=m[0] and m2[1]>=m[1] and m[2]!=m2[2]:
                                confilict = True
                                break
                            ## m2 fully inside m, another id
                            if m[0]<=m2[0] and m[1]>=m2[1] and m[2]!=m2[2]:
                                continue
                            if m[3]<m2[3]:
                                confilict = True
                                break
                        '''
                        if m[1]>m2[0] and m[0]<m2[0] and m[1]<m2[1] and m[3]<m2[3]:
                            confilict = True
#                            print "0"
                            break
                        ## m2 fully inside m, same id, better score
                        if m[0]<=m2[0] and m[1]>=m2[1] and m[2]==m2[2] and m[3]<m2[3]:
                            confilict = True
#                            print "1"
                            break
                        ## m2 fully covers m, another id
                        if m2[0]<=m[0] and m2[1]>=m[1] and m[2]!=m2[2]:
                            confilict = True
                            #print "2"
                            break
                        '''
                    if not confilict:
                        final.append(m)
            return final
            

	def process_text_fast(self, text, threshold=0.5, filter_overlap=False):
            sents = [text] #text.split(".")
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

	def process_text(self, text, threshold=0.5, filter_overlap=False):
            #'''
                return self.process_text_fast_new(text, threshold, filter_overlap)
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
            #'''

	def __init__(self, model):
            self.model = model
	#def __init__(self, repdir=None, ant=None, datadir=None, addNull=False):
        #    self.rd = reader.Reader(open("data/hp.obo"), False)
        #    self.model = phrase_model.NCRModel(Config(), self.rd)
        #    self.model.load_params(repdir)
            '''
            if ant == None:
                self.ant = phrase_annotator.create_annotator(repdir, datadir, True, addNull)
            else:
                self.ant = ant
            '''



def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--repdir', help="The location where the checkpoints are stored, default is \'checkpoints/\'", default="checkpoints")
	parser.add_argument('--input_dir')
	parser.add_argument('--input')
	parser.add_argument('--output_dir')
	parser.add_argument('--threshold', type=float, default=0.5)
	parser.add_argument('--filter_overlap', action='store_true', default=True)
	args = parser.parse_args()

	sys.stderr.write("Initializing NCR...\n")
	textAnt = TextAnnotator(args.repdir, datadir="data/", addNull=True)
	sys.stderr.write("Done.\n")
	#sent_accuracy.find_sent_accuracy(lambda text: [x[2] for x in textAnt.process_text(text, 0.3, True )], 'labeled_sentences.p', textAnt.ant.rd)
	#exit()

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
			print "["+str(res[0])+"::"+str(res[1])+"]\t" , res[2], "|", text[res[0]:res[1]], "\t", res[3], "\t", textAnt.rd.names[res[2]]
		print "Time elapsed: "+ ("%.2f" % (end_time-start_time)) + "s"
		if args.input is not None:
			break


if __name__ == '__main__':
	main()
