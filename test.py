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
    chunks=chunck_NP.extract_NP_fromText(text)
    for sentence in chunks:
        for np in sentence:
            input_term=" ".join(np)
            res, sc =ant.get_hp_id(input_term)
            if sc > 0.5:
				real_id_tokens=res.split(":")
				real_id=real_id_tokens[0]+"_"+real_id_tokens[1]
				output_file.write(input_term + "\t" + real_id+"\n")



def main():
	ps=argparse.ArgumentParser()
	ps.add_argument('input_dir', type=str)
	ps.add_argument('output_dir', type=str)
	args=ps.parse_args()

	global ant
	ant = process_querry.NeuralAnnotator()

	data=[]
	for text_file in os.listdir(args.input_dir):
		text=open(args.input_dir+"/"+text_file).read().decode('utf8').encode('ascii','ignore')
		output_addr=args.output_dir+"/"+text_file
		data.append([text,output_addr])
		process_file((text,output_addr))
	print "data tuple created..."

	pool=Pool(20)
	pool.map(process_file,tuple(data))


if __name__ == '__main__':
	main()
