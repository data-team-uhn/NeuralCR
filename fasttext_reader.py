from random import shuffle
import random
import numpy as np
import cPickle as pickle
from itertools import product
import h5py
import fasttext

total=0
marked={}
'''
    def phrase2vec(self, phrase):
	tokens = phrase.lower().replace(',',' , ').replace('-',' ')\
                .replace(';', ' ; ').replace('/', ' / ').replace('(', ' ( ')\
                .replace(')', ' ) ').replace('.', ' . ').strip().split()
	tokens = ["INT" if w.isdigit() else ("FLOAT" if is_number(w) else w) for w in tmp]
	seq = np.array( [self.word_model[w] for w in tokens] ) 
'''


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def tokenize(phrase):
	tmp = phrase.lower().replace(',',' , ').replace('-',' ').replace(';', ' ; ').replace('/', ' / ').replace('(', ' ( ').replace(')', ' ) ').replace('.', ' . ').strip().split()
	return ["INT" if w.isdigit() else ("FLOAT" if is_number(w) else w) for w in tmp]

def dfs(c, kids, mark):
	mark.add(c)
	for kid in kids[c]:
		if kid not in mark:
			dfs(kid, kids, mark)

def read_oboFile(oboFile, topid=None):
	names={}
	def_text={}
	kids={}
	parents={}
	real_id = {}
	while True:
		line=oboFile.readline()
		if line == "":
			break
		tokens=line.strip().split(" ")
		if tokens[0]=="id:":
			hp_id=tokens[1]
			parents[hp_id] = []
			kids[hp_id] = []
			names[hp_id] = []
			real_id[hp_id] = hp_id

		if tokens[0] == "def:":
			def_text[hp_id] = [line[line.index("\"")+1:line.rindex("\"")]]
		if tokens[0] == "name:":
			names[hp_id] = [' '.join(tokens[1:])]
		if tokens[0] == "synonym:":
			last_index = (i for i,v in enumerate(tokens) if v.endswith("\"")).next()
			names[hp_id].append( ' '.join(tokens[1:last_index+ 1]).strip("\"") )
		if tokens[0] == "alt_id:":
			real_id[tokens[1]] = hp_id
	
	oboFile.seek(0)
	while True:
		line=oboFile.readline()
		if line == "":
			break
		tokens=line.strip().split(" ")
		if tokens[0]=="id:":
			hp_id=tokens[1]

		if tokens[0]=="is_a:":
			kids[tokens[1]].append(hp_id)
			parents[hp_id].append(tokens[1])
	mark=set()
	dfs(topid, kids, mark)
	names = {c:names[c] for c in mark}
	def_text = {c:def_text[c] for c in mark if c in def_text}
	parents = {c:parents[c] for c in mark}
	kids = {c:kids[c] for c in mark}
	for c in parents:
		parents[c]=[p for p in parents[c] if p in mark]
	total_names = []
	for c in names:
		for name in names[c]:
			total_names.append(name)
			#print name
#	print len(total_names)
	return names, kids, parents, real_id, def_text

def create_negatives(text, num):
    neg_tokens = tokenize(text)

    indecies = np.random.choice(len(neg_tokens), num)
    lengths = np.random.randint(1, 10, num)

    negative_phrases = [' '.join(neg_tokens[indecies[i]:indecies[i]+lengths[i]])
                                for i in range(num)]
    return negative_phrases


class Reader:
	def phrase2ids(self, phrase, add_words=False):
		#		if len(tokenize(phrase)) > self.max_length:
		#	print len(tokenize(phrase))
		tokens = tokenize(phrase)
		tokens = tokens[:self.max_length-1]
		tokens.append("<END-TOKEN>")
		if add_words:
			for w in tokens:
				if w not in self.word2id:
					self.word2id[w] = len(self.word2id)

		ids = np.array( [self.word2id[w] if w in self.word2id else self.word2id[self.unkown_term] for w in tokens] )

		return ids #, None



	def phrase2vec(self, phrase):
		tokens = tokenize(phrase)[:self.max_length-1]
		return np.array( [self.word_model[w] for w in tokens] ) 


	def _update_ancestry(self, c):
		cid = self.concept2id[c]
		if np.sum(self.ancestry_mask[cid]) > 0:
			return self.ancestry_mask[cid]

		self.ancestry_mask[cid,cid] = 1.0

		for p in self.parents[c]:
			self.ancestry_mask[cid, self.concept2id[p]] = 1.0
			self.ancestry_mask[cid,:]=np.maximum(self.ancestry_mask[cid,:], self._update_ancestry(p))

		return self.ancestry_mask[cid,:]

	def _update_ancestry_sparse(self, c):
		cid = self.concept2id[c]
		if cid in self.ancestrs:
			return self.ancestrs[cid]

                self.ancestrs[cid] = set([cid])

		for p in self.parents[c]:
                    self.ancestrs[cid].update(self._update_ancestry_sparse(p))
                return self.ancestrs[cid]




	def __init__(self, data_dir='data/', include_negs=False):
                self.include_negs = include_negs
		## Create word to id
		self.max_length = 50 #max([len(s[0]) for s in self.samples])
		self.word_embed_size = 100

                

		print "loading word model..."
		self.word_model = fasttext.load_model(data_dir+'/model_pmc.bin')

		###################### Read HPO ######################
		self.names, self.kids, self.parents, self.real_id, self.def_text = read_oboFile(open(data_dir+'/hp.obo'), "HP:0000118")
		self.concepts = [c for c in self.names.keys()]
		self.concept2id = dict(zip(self.concepts,range(len(self.concepts))))

		self.name2conceptid = {}
		for c in self.concepts:
			for name in self.names[c]:
				normalized_name = name.strip().lower()
				self.name2conceptid[normalized_name] = self.concept2id[c]

		self.ancestry_mask = np.zeros((len(self.concepts), len(self.concepts)))
                self.ancestrs = {}
		self.samples = []
		for c in self.concepts:
			self._update_ancestry(c)
			self._update_ancestry_sparse(c)
			for name in self.names[c]:
				self.samples.append((self.phrase2vec(name), [self.concept2id[c]], 'name')) 
			'''
			if c in self.def_text:
				for def_text in self.def_text[c]:
					self.samples.append( (self.phrase2ids(def_text, True), [self.concept2id[c]], 'def_text')) 
			'''

		'''
		self.samples_by_concept = []
		for i,c in enumerate(self.concepts):
			tmp_sample = {}
			raw_seq = [self.phrase2ids(name) for name in self.names[c]]
			tmp_sample['seq_len'] = np.array([len(s) for s in raw_seq])
			tmp_sample['seq'] = np.zeros((len(raw_seq), self.max_length), dtype = int)
			for j,s in enumerate(raw_seq):
				tmp_sample['seq'][j,:tmp_sample['seq_len'][j]] = s
			self.samples_by_concept.append(tmp_sample)

		self.word2id[self.unkown_term] = len(self.word2id)
		self.word2vec = np.vstack((self.word2vec, np.zeros((len(self.word2id) - initial_word2id_size,self.word2vec.shape[1]))))
		'''
                self.sparse_ancestrs = []
                for cid in self.ancestrs:
                    #self.sparse_ancestrs += [[ancid, cid] for ancid in self.ancestrs[cid]]
                    self.sparse_ancestrs += [[cid, ancid] for ancid in self.ancestrs[cid]]


                if self.include_negs:
                    '''
                    ub_names, ub_kids, ub_parents, ub_real_id, ub_def_text = read_oboFile(open('data/uberon.obo'), "UBERON:0000062")
                    self.ub_negs = [x for concept in ub_names for x in ub_names[concept]]
                    ub_names2, ub_kids, ub_parents, ub_real_id, ub_def_text = read_oboFile(open('data/uberon.obo'), "UBERON:0000064")
                    self.ub_negs += [x for concept in ub_names2 for x in ub_names2[concept]]
                    '''
                    none_id = len(self.concepts)
                    self.concepts.append('HP:None')
                    self.names['HP:None']=['None']
                    
                    wiki_text = open(data_dir+'/wiki_text').read()
                    raw_negs = create_negatives(wiki_text[:10000000], 10000)
                    #raw_negs += self.ub_negs

                    self.neg_samples = []
                    self.sparse_ancestrs.append([none_id, none_id])
                    for x in raw_negs:
                        self.neg_samples.append((self.phrase2vec(x), [none_id], 'neg'))

                    self.samples += self.neg_samples





		self.reset_counter()
                print "word model loaded"

	def reset_counter(self):
#		self.mixed_samples =  self.pmc_samples
		self.mixed_samples = self.samples #+ self.pmc_samples + self.wiki_samples 
		shuffle(self.mixed_samples)
		self.counter = 0

	def create_test_sample(self, phrases):
		seq = np.zeros((len(phrases), self.max_length, self.word_embed_size), dtype = float)
		phrase_vecs = [self.phrase2vec(phrase) for phrase in phrases]
		seq_lengths = np.array([len(phrase) for phrase in phrase_vecs])
		for i,s in enumerate(phrase_vecs):
			seq[i,:seq_lengths[i]] = s
		return {'seq':seq, 'seq_len':seq_lengths}

	def read_batch(self, batch_size):#, compare_size):
		if self.counter >= len(self.mixed_samples):
			return None
		ending = min(len(self.mixed_samples), self.counter + batch_size)
		raw_batch = self.mixed_samples[self.counter : ending]

		sequence_lengths = np.array([len(s[0]) for s in raw_batch])
		sequences = np.zeros((min(batch_size, ending-self.counter), self.max_length, self.word_embed_size), dtype = float)
		for i,s in enumerate(raw_batch):
			sequences[i,:sequence_lengths[i]] = s[0]
		hpo_id = np.array([s[1][0] if len(s[1])>0 else 0 for s in raw_batch])

                type_to_id = {'name':0, 'def_text':1, 'neg':2}
		type_id = np.array([type_to_id[s[2]] for s in raw_batch])

		self.counter = ending

		return {'seq':sequences, 'seq_len':sequence_lengths, 'hp_id':hpo_id, 'type_id':type_id} #, 'comparables':comparables, 'comparables_mask':comparables_mask}

def main():
	'''
	oboFile=open("data/uberon.obo")
	read_oboFile(oboFile, "UBERON:0010000")
	return
	'''
	oboFile=open("data/hp.obo")
	vectorFile=open("data/vectors.txt")
#        vectorFile=open("train_data_gen/test_vectors.txt")
	reader = Reader(oboFile, True)
        print reader.concept2id['HP:0000246']
        exit()
        print reader.read_batch(10)
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

