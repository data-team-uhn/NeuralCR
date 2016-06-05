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
	tmp = phrase.lower().replace(',',' ').replace('-',' ').replace(';', ' ').replace('/', ' / ').replace('(', ' ( ').replace(')', ' ) ').strip().split()
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

		if tokens[0] == "name:":
			names[hp_id] = [' '.join(tokens[1:])]
		if tokens[0] == "synonym:":
			last_index = (i for i,v in enumerate(tokens) if v.endswith("\"")).next()
			names[hp_id].append( ' '.join(tokens[1:last_index+ 1]).strip("\"") )
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
	return names, kids, parents, top_nodes


def store_data(data, directory):
	types = [ 'triplets' , 'labels', 'raw' ]
	for x in data:
		for i,t in enumerate(types):
			address = directory + "/" + x + "_" + t + ".npy"
			if t != 'raw':
				print address, data[x][i].shape
				np.save(address, data[x][i])



class Reader:

	def _update_ancestry(self, c):
		cid = self.concept2id[c]
		if np.sum(self.ancestry_mask[cid]) > 0:
			return self.ancestry_mask[cid]

		self.ancestry_mask[cid,cid] = 1.0

		for p in self.parents[c]:
			self.ancestry_mask[cid, self.concept2id[p]] = 1.0
			self.ancestry_mask[cid,:]=np.maximum(self.ancestry_mask[cid,:], self._update_ancestry(p))

		return self.ancestry_mask[cid,:]

	def phrase2ids(self, phrase):
		tokens = tokenize(phrase)
		tokens.append("<END-TOKEN>")
		for w in tokens:
			if w not in self.word2id:
				self.word2id[w] = len(self.word2id)
		ids = np.array( [self.word2id[w] for w in tokens] )
		return ids

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
		word_vectors=[]
		self.word2id={}
		for i,line in enumerate(vectorFile):
			tokens = line.strip().split(" ")
			word_vectors.append(np.array(map(float,tokens[1:])))
			self.word2id[tokens[0]] = i
		self.word2vec = np.vstack(word_vectors)                
		initial_word2id_size = len(self.word2id)
		## Create concept to id
		self.names, self.kids, self.parents, self.top_nodes = read_oboFile(oboFile)
		self.concepts = [c for c in self.names.keys()]
		self.concept2id = dict(zip(self.concepts,range(len(self.concepts))))
		self.concept_id_list = set(self.concept2id.values())
		self.top_nodes_id_list = set([self.concept2id[x] for x in self.top_nodes])

		self.ancestry_mask = np.zeros((len(self.concepts), len(self.concepts)))
		self.ancestry_list = {}
		self.samples = []
		for c in self.concepts:
			self._update_ancestry(c)
			self._update_ancestry_list(c)
			for name in self.names[c]:
				self.samples.append( (self.phrase2ids(name), self.concept2id[c] )) #self.ancestry_mask[self.concept2id[c],:]))

		self.word2vec = np.vstack((self.word2vec, np.zeros((len(self.word2id) - initial_word2id_size,self.word2vec.shape[1]))))

		self.reset_counter()
		self.max_length = max([len(s[0]) for s in self.samples])
		print self.max_length

	def init_pmc_data(self, pmcFile):
		pmc_raw = pickle.load(pmcFile)
		print len(pmc_raw)


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
		shuffle(self.samples)
		self.counter = 0

	def create_test_sample(self, phrases):
		seq = np.zeros((len(phrases), self.max_length), dtype = int)
		phrase_ids = [self.phrase2ids(phrase) for phrase in phrases]
		seq_lengths = np.array([len(phrase) for phrase in phrase_ids])
		for i,s in enumerate(phrase_ids):
			seq[i,:seq_lengths[i]] = s
		return seq, seq_lengths


	def read_batch_from_hpo(self, batch_size):
		if self.counter >= len(self.samples):
			return None
		ending = min(len(self.samples), self.counter + batch_size)
		raw_batch = self.samples[self.counter : ending]

		sequence_lengths = np.array([len(s[0]) for s in raw_batch])
		sequences = np.zeros((min(batch_size, ending-self.counter), self.max_length), dtype = int)
		for i,s in enumerate(raw_batch):
			sequences[i,:sequence_lengths[i]] = s[0]
#		ancestry_mask = np.vstack([s[1] for s in raw_batch])
		hpo_ids = np.array([s[1] for s in raw_batch])

		self.counter = ending

		#return (sequences, sequence_lengths, ancestry_mask)
		return (sequences, sequence_lengths, hpo_ids)



	def read_batch(self, batch_size, compare_size):
		if self.counter >= len(self.samples):
			return None
		ending = min(len(self.samples), self.counter + batch_size)
		raw_batch = self.samples[self.counter : ending]

		sequence_lengths = np.array([len(s[0]) for s in raw_batch])
		sequences = np.zeros((min(batch_size, ending-self.counter), self.max_length), dtype = int)
		comparables = np.zeros((min(batch_size, ending-self.counter), compare_size), dtype = int)
		comparables_mask = np.zeros((min(batch_size, ending-self.counter), compare_size), dtype = int)
		'''
		comparables = np.zeros((min(batch_size, ending-self.counter), len(self.concept_id_list)), dtype = int)
		comparables_mask = np.zeros((min(batch_size, ending-self.counter), len(self.concept_id_list)), dtype = float)
		'''
		for i,s in enumerate(raw_batch):
			sequences[i,:sequence_lengths[i]] = s[0]

			tmp_comp = list(self.ancestry_list[s[1]]) + list(self.top_nodes_id_list - self.ancestry_list[s[1]])
			tmp_comp += list(random.sample(self.concept_id_list - self.top_nodes_id_list - self.ancestry_list[s[1]], compare_size-len(tmp_comp)))
#			tmp_comp = list(self.ancestry_list[s[1]]) + list(self.concept_id_list - self.ancestry_list[s[1]])
			tmp_comp_mask = [1]*len(self.ancestry_list[s[1]]) + [0]*(compare_size-len(self.ancestry_list[s[1]]))
#			tmp_comp_mask = [1]*len(self.ancestry_list[s[1]]) + [0]*(len(self.concept_id_list)-len(self.ancestry_list[s[1]]))

#			tmp_comp = list(self.concept_id_list)
#			tmp_comp_mask = self.ancestry_mask[s[1],:]
			comparables[i,:] = np.array(tmp_comp)
			comparables_mask[i,:] = np.array(tmp_comp_mask)

#		ancestry_mask = np.vstack([s[1] for s in raw_batch])
		hpo_ids = np.array([s[1] for s in raw_batch])

		self.counter = ending

		#return (sequences, sequence_lengths, ancestry_mask)
		return (sequences, sequence_lengths, hpo_ids, comparables, comparables_mask)
		'''
		if self.batch_counter > len(self.batches):
			return None
		cur_batch = self.batches[self.batch_counter]
		self.batch_counter = self.batch_counter+1
		return cur_batch
		'''
def main():

	oboFile=open("hp.obo")
	vectorFile=open("vectors.txt")
#        vectorFile=open("train_data_gen/test_vectors.txt")
	reader = Reader(oboFile, vectorFile)
	print len(reader.top_nodes_id_list)
	batch = reader.read_batch(1, 300)
	print batch
#	print batch[0], reader.


	'''
        for i in range(100):
            print reader.read_batch()
	'''
#	reader.init_pmc_data(open('pmc_raw.p'))

if __name__ == "__main__":
	main()

