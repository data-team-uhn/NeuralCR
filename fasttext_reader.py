from random import shuffle
import random
import numpy as np
import cPickle as pickle
from itertools import product
import h5py
import fasttext
from onto import Ontology

total=0
marked={}

def create_negatives(text, num):
    neg_tokens = tokenize(text)

    indecies = np.random.choice(len(neg_tokens), num)
    lengths = np.random.randint(1, 10, num)

    negative_phrases = [' '.join(neg_tokens[indecies[i]:indecies[i]+lengths[i]])
                                for i in range(num)]
    return negative_phrases


class Reader:
    def __init__(self, ont, neg_samples=None): #data_dir='data/', include_negs=False):
        ## Create word to id
        self.word_embed_size = 100

        

        print "loading word model..."
        self.ont = ont

        ###################### Read HPO ######################
        self.samples = []
        for c in self.ont.concepts:
                for name in self.ont.names[c]:
                        self.samples.append([name, [self.ont.concept2id[c]], 'name']) 

        self.neg_samples = []
        if neg_samples!=None:
            none_id = len(self.ont.concepts)
            self.neg_samples = [[x, [none_id], 'neg'] for x in neg_samples]
        self.reset_counter()
        print "word model loaded"

    def reset_counter(self):
#		self.mixed_samples =  self.pmc_samples
        self.mixed_samples = self.samples + self.neg_samples#+ self.pmc_samples + self.wiki_samples 
        shuffle(self.mixed_samples)
        self.counter = 0

    def read_batch(self, batch_size):#, compare_size):
        if self.counter >= len(self.mixed_samples):
            return None
        ending = min(len(self.mixed_samples), self.counter + batch_size)
        raw_batch = self.mixed_samples[self.counter : ending]

        return raw_batch

def main():
	'''
	oboFile=open("data/uberon.obo")
	read_oboFile(oboFile, "UBERON:0010000")
	return
	'''
	oboFile=open("data/hp.obo")
	vectorFile=open("data/vectors.txt")
#        vectorFile=open("train_data_gen/test_vectors.txt")
	reader = Reader(Ontology())
	#reader = Reader(oboFile, True)
        print reader.ont.concept2id['HP:0000246']
        print reader.read_batch(10)
        exit()
        return
        print (reader.sparse_ancestrs)
        print len(reader.sparse_ancestrs)
        print np.sum(reader.ancestry_mask)
        exit()
	epoch = 0
	while True:
		print  epoch
		batch = reader.read_batch(64)
		print batch
		if batch == None:
			break
		epoch += 1
	return
	print batch['seq'].shape
	print batch['seq_len'].shape
	print batch['hp_id'].shape

	return
	print reader.text_def["HP:0000118"]
	return
	reader.init_uberon_list()
	reader.reset_counter()
	batch = reader.read_batch(10)
	print batch
	return
	#reader.reset_counter()

	reader.init_pmc_data(open('data/pmc_samples.p'),open('data/pmc_id2text.p'), open('data/pmc_labels.p'))
	reader.init_wiki_data(open('data/wiki-samples.p'))
	print "inited"
	reader.reset_counter()
	'''
	while True:
		batch = reader.read_batch(128, 300)
		if batch == None:
			exit()
	'''

	batch = reader.read_batch(4)
	print batch

	print len(reader.wiki_raws)
	print len(reader.samples)
	print len(reader.wiki_samples)
	print len(reader.pmc_samples)
	print len(reader.mixed_samples)
#	print batch[0], reader.


	'''
        for i in range(100):
            print reader.read_batch()
	'''

if __name__ == "__main__":
	main()

