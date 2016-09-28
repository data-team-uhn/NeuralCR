from random import shuffle
import random
import numpy as np
import cPickle as pickle
from itertools import product

total=0
marked={}

def bfs(start, kids, upper):
	que=[start]
	d={}
	d[start]=0
	visited=set()
	visited.add(start)
	tail=0
	local=[]
	while tail < len(que):
		v=que[tail]
		if d[v] >= upper:
			break
		for u in kids[v]:
			if u not in visited:
				d[u]=d[v]+1
				que.append(u)
				visited.add(u)
		tail+=1
	return visited

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def tokenize(phrase):
	tmp = phrase.lower().replace(',',' , ').replace('-',' ').replace(';', ' ; ').replace('/', ' / ').replace('(', ' ( ').replace(')', ' ) ').strip().split()
	return ["INT" if w.isdigit() else ("FLOAT" if is_number(w) else w) for w in tmp]

def dfs(c, kids, mark):
	mark.add(c)
	for kid in kids[c]:
		if kid not in mark:
			dfs(kid, kids, mark)

def read_oboFile(oboFile):
	names={}
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
	dfs("HP:0000118", kids, mark)
	top_nodes = bfs("HP:0000118", kids, 2)
	names = {c:names[c] for c in mark}
	parents = {c:parents[c] for c in mark}
	kids = {c:kids[c] for c in mark}
	for c in parents:
		parents[c]=[p for p in parents[c] if p in mark]
	return names, kids, parents, top_nodes, real_id


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

		return ids, None

	def _update_ancestry(self, c):
		cid = self.concept2id[c]
		if np.sum(self.ancestry_mask[cid]) > 0:
			return self.ancestry_mask[cid]

		self.ancestry_mask[cid,cid] = 1.0

		for p in self.parents[c]:
			self.ancestry_mask[cid, self.concept2id[p]] = 1.0
			self.ancestry_mask[cid,:]=np.maximum(self.ancestry_mask[cid,:], self._update_ancestry(p))

		return self.ancestry_mask[cid,:]

	def _update_ancestry_list(self,c):
		cid = self.concept2id[c]
		if cid in self.ancestry_list:
			return self.ancestry_list[cid]

		self.ancestry_list[cid] = set([cid])

		for p in self.parents[c]:
			self.ancestry_list[cid].update(self._update_ancestry_list(p))

		return self.ancestry_list[cid]



	def __init__(self, oboFile, vectorFile):
		## Create word to id
		self.max_length = 50 #max([len(s[0]) for s in self.samples])
		word_vectors=[]
		self.word2id={}
		for i,line in enumerate(vectorFile):
			tokens = line.strip().split(" ")
			word_vectors.append(np.array(map(float,tokens[1:])))
			self.word2id[tokens[0]] = i
		self.word2vec = np.vstack(word_vectors)


		initial_word2id_size = len(self.word2id)
		## Create concept to id
		self.names, self.kids, self.parents, self.top_nodes, self.real_id = read_oboFile(oboFile)

		self.concepts = [c for c in self.names.keys()]
		self.concept2id = dict(zip(self.concepts,range(len(self.concepts))))
		self.concept_id_list = set(self.concept2id.values())
		self.kids_id = {self.concept2id[c]:set([self.concept2id[cc] for cc in self.kids[c]]) for c in self.kids}
		self.unkown_term = "<UNKNOWN>"

		self.name2conceptid = {}
		for c in self.concepts:
			for name in self.names[c]:
				normalized_name = name.strip().lower()
				self.name2conceptid[normalized_name] = self.concept2id[c]

		self.ancestry_list = {}
		self.ancestry_mask = np.zeros((len(self.concepts), len(self.concepts)))
		self.samples = []
		for c in self.concepts:
			self._update_ancestry(c)
			self._update_ancestry_list(c)
			for name in self.names[c]:
				self.samples.append( (self.phrase2ids(name, True), [self.concept2id[c]] )) 
		self.word2id[self.unkown_term] = len(self.word2id)

		self.word2vec = np.vstack((self.word2vec, np.zeros((len(self.word2id) - initial_word2id_size,self.word2vec.shape[1]))))

		self.pmc_has_init = False
		self.wiki_has_init = False
		self.reset_counter()

	def reset_wiki_reader(self):
		self.wiki_samples = []

		if not self.wiki_has_init:
			return
		shuffle(self.wiki_raws)
		for i in range(10000):
			tokens = self.phrase2ids(self.wiki_raws[i])
			if len(tokens[0])>=self.max_length:
				continue
			self.wiki_samples.append((tokens, []))

	def reset_pmc_reader(self):
		self.pmc_samples=[]
		if not self.pmc_has_init:
			return
		for c in self.concepts:
			for name in self.names[c]:
				normalized_name = name.strip().lower()
				if normalized_name in self.pmc_raws:
					for i in range(2):
						buck = random.sample(self.pmc_raws[normalized_name],1)[0]
						textid = random.sample(self.pmc_raws[normalized_name][buck],1)[0]

						text = self.pmc_id2text[textid]
						tokens = self.phrase2ids(text)
						if len(tokens[0])>=self.max_length:
							continue
						self.pmc_samples.append((tokens, [self.name2conceptid[x] for x in self.textid2labels[textid]]))

	def init_wiki_data(self, wikiFile):
		self.wiki_has_init = True
		self.wiki_raws = pickle.load(wikiFile)

	def init_pmc_data(self, pmcFile, pmcid2textFile, pmclabelsFile):
		self.pmc_has_init = True
		self.pmc_raws = pickle.load(pmcFile)
#		print self.pmc_raws.keys()
		self.pmc_id2text = pickle.load(pmcid2textFile)
		self.textid2labels = pickle.load(pmclabelsFile)
		self.reset_pmc_reader()
	#	print pmc_samples


		## Create buckets and samples
		'''
		samples = []
		sizes = []
		self.bucket = {i:[] for i in range(1,30)} ## 20?
		for c in concepts:
			for name in self.names[c]:
                        samples.append( [self.phrase2ids(name), self.ancestry_mask[self.concept2id[c],:]] )
                        self.bucket[samples[-1][0].shape[0]].append( (self.phrase2ids(name), self.ancestry_mask[self.concept2id[c],:]) )
                for i in self.bucket:
                    shuffle(self.bucket[i])
                self.batches = []
                for i in self.bucket:
                    counter = 0
                    while counter < len(self.bucket[i]):
                        entry = [ np.vstack( [x[index] for x in self.bucket[i][counter:min(len(self.bucket[i]),counter+batch_size)]]) for index in [0,1] ]
                        self.batches.append(entry)
                        counter += batch_size
#                    print i, len(self.bucket[i])
                self.batch_counter = 0
                shuffle(self.batches)

		self.word2vec = np.vstack((self.word2vec, np.zeros((len(self.word2id) - initial_word2id_size,self.word2vec.shape[1]))))
		'''
	def reset_counter(self):
		self.reset_pmc_reader()
		self.reset_wiki_reader()
#		self.mixed_samples =  self.pmc_samples
		self.mixed_samples = self.samples + self.pmc_samples + self.wiki_samples
		shuffle(self.mixed_samples)
		self.counter = 0

	def create_test_sample(self, phrases):
		seq = np.zeros((len(phrases), self.max_length), dtype = int)
		phrase_ids = [self.phrase2ids(phrase) for phrase in phrases]
		seq_lengths = np.array([len(phrase[0]) for phrase in phrase_ids])
		for i,s in enumerate(phrase_ids):
			seq[i,:seq_lengths[i]] = s[0]
		return {'seq':seq, 'seq_len':seq_lengths}


	def read_batch(self, batch_size):#, compare_size):
		if self.counter >= len(self.mixed_samples):
			return None
		ending = min(len(self.mixed_samples), self.counter + batch_size)
		raw_batch = self.mixed_samples[self.counter : ending]

		sequence_lengths = np.array([len(s[0][0]) for s in raw_batch])
		sequences = np.zeros((min(batch_size, ending-self.counter), self.max_length), dtype = int)
#		comparables = np.zeros((min(batch_size, ending-self.counter), compare_size), dtype = int)
#		comparables_mask = np.zeros((min(batch_size, ending-self.counter), compare_size), dtype = int)
		for i,s in enumerate(raw_batch):
			sequences[i,:sequence_lengths[i]], _ = s[0]
			ancestrs = set()
			all_kids = set()
			for hit in s[1]:
				ancestrs.update(self.ancestry_list[hit])
			for ancestr in ancestrs:
				all_kids.update(self.kids_id[ancestr])

			all_positives = ancestrs
			selected_positives = ancestrs

#			selected_kids = random.sample(all_kids - all_positives, min(compare_size - len(selected_positives), len(all_kids-all_positives)))

#			tmp_comp = list(selected_positives) + list(selected_kids)
#			tmp_comp += list(random.sample(self.concept_id_list - set(tmp_comp)- all_positives, compare_size-len(tmp_comp)))

#			tmp_comp_mask = [1]*len(selected_positives) + [0]*(compare_size-len(selected_positives))

#			comparables[i,:] = np.array(tmp_comp)
#			comparables_mask[i,:] = np.array(tmp_comp_mask)

		hpo_ids = np.array([s[1][0] if len(s[1])>0 else 0 for s in raw_batch])

		self.counter = ending

		return {'seq':sequences, 'seq_len':sequence_lengths, 'hp_id':hpo_ids} #, 'comparables':comparables, 'comparables_mask':comparables_mask}

def main():
	oboFile=open("data/hp.obo")
	vectorFile=open("data/vectors.txt")
#        vectorFile=open("train_data_gen/test_vectors.txt")
	reader = Reader(oboFile, vectorFile)
	#batch = reader.read_batch(5)
	#print batch
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

