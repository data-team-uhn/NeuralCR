import process_querry
from multiprocessing import Pool
import argparse
import sys
import chunck_NP
import os
import nltk

ant = None

def process_file((text,output_addr)):
    output_file=open(output_addr,"w")
    #chunks=chunck_NP.extract_NP_fromText(text)
#    for sentence in chunks:
 
    max_phrase_size = 7
    tokenized_text = chunck_NP.tokenize(text) #text.strip().split()
    hpo_terms = set()

    for i,word in enumerate(tokenized_text):
        phrase=""
        for j in range(min(i,max_phrase_size)):
            phrase= tokenized_text[i-j] + " " + phrase
      #      print phrase
            res, sc =ant.get_hp_id(phrase)
            if sc > 0.7:
                real_id_tokens=res.split(":")
                real_id=real_id_tokens[0]+"_"+real_id_tokens[1]
               # print(phrase + "\t" + real_id + '\t' + str(sc) +"\n")
                hpo_terms.add(real_id)
#		output_file.write(real_id+"\n")
        #print "--"
    for term in hpo_terms:
	output_file.write(term+"\n")

def main():
	ps=argparse.ArgumentParser()
	ps.add_argument('input_dir', type=str)
	ps.add_argument('output_dir', type=str)
	args = ps.parse_args()

	global ant
        print "init started"
	ant = process_querry.NeuralAnnotator()
        print "init done"

	data=[]
	for text_file in os.listdir(args.input_dir):
		text = open(args.input_dir+"/"+text_file).read().decode('utf8').encode('ascii','ignore')
		output_addr = args.output_dir+"/"+text_file
		data.append([text,output_addr])
                print text_file
		process_file((text,output_addr))
	print "data tuple created..."

        '''
	pool = Pool(5)
	pool.map(process_file,tuple(data))
        pool.join()
        '''


if __name__ == '__main__':
	main()
