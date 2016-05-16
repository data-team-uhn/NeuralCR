from random import shuffle
import random
import numpy as np
import cPickle as pickle
from itertools import product

total=0
marked={}

def bfs(g, start, lower, upper):
	que=[start]
	d={}
	d[start]=0
	visited={}
	visited[start]=True
	tail=0
	local=[]
	while tail < len(que):
		v=que[tail]
		if d[v] > upper:
			break
		for u in g[v]:
			if u not in visited:
				d[u]=d[v]+1
				que.append(u)
				if d[u]>=lower:
					local.append(u)
				visited[u]=True
		tail+=1
	return local

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def randomize_triplet(t):
	new_t = [x for x in t] #.append(0)
	new_t.append(np.array([1.0, 0.0]))
	if random.random() > 0.5:
		new_t[1], new_t[2] = new_t[2], new_t[1]
		new_t[3] = np.array([0.0, 1.0])
	return new_t

def tokenize(phrase):
	tmp = phrase.lower().replace(',',' ').replace('-',' ').replace(';', ' ').replace('/', ' / ').replace('(', ' ( ').replace(')', ' ) ').strip().split()
	return ["INT" if w.isdigit() else ("FLOAT" if is_number(w) else w) for w in tmp]

def embed_phrase(phrase, wordList, padded_size):
	words = tokenize(phrase)
	em_list = [wordList[w] for w in words]
	em_list += [np.zeros(wordList["the"].shape)]*(padded_size-len(words))
	embeding = np.concatenate(em_list)
	return embeding

def check_phrase(phrase, wordList, word_limit):
	words = tokenize(phrase)
	if len(words) <= word_limit and all([(w in wordList) for w in words]):
		return True
	return False

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
	names = {c:names[c] for c in mark}
	parents = {c:parents[c] for c in mark}
	kids = {c:kids[c] for c in mark}
	for c in parents:
		parents[c]=[p for p in parents[c] if p in mark]
	return names, kids, parents



def postprocess_triplets(triplets, wordVector, word_limit):
	shuffle(triplets)
	filtered_triplets = [triplet for triplet in triplets if all([check_phrase(phrase, wordVector, word_limit) for phrase in triplet])]
	vectorized_triplets = [ [embed_phrase(phrase, wordVector, word_limit) for phrase in triplet] for triplet in filtered_triplets ]
	randomized_triplets = map( randomize_triplet, vectorized_triplets )
	return np.stack ([np.stack([t[0] for t in randomized_triplets]), np.stack([t[1] for t in randomized_triplets]), np.stack([t[2] for t in randomized_triplets])], axis=2), np.stack([t[3] for t in randomized_triplets]), filtered_triplets

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
		for w in tokens:
			if w not in self.word2id:
				self.word2id[w] = len(self.word2id)
		ids = np.array( [self.word2id[w] for w in tokens] )
		return ids


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
		self.names, self.kids, self.parents = read_oboFile(oboFile)
		self.concepts = [c for c in self.names.keys()]
		self.concept2id = dict(zip(self.concepts,range(len(self.concepts))))

		self.ancestry_mask = np.zeros((len(self.concepts), len(self.concepts)))
		self.samples = []
		for c in self.concepts:
			self._update_ancestry(c)
			for name in self.names[c]:
				self.samples.append( (self.phrase2ids(name), self.concept2id[c] )) #self.ancestry_mask[self.concept2id[c],:]))

		self.word2vec = np.vstack((self.word2vec, np.zeros((len(self.word2id) - initial_word2id_size,self.word2vec.shape[1]))))

		self.reset_counter()
		self.max_length = max([len(s[0]) for s in self.samples])
		print self.max_length


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

	def read_batch(self, batch_size):
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
	reader = Reader(oboFile, vectorFile, 10)
        for i in range(100):
            print reader.read_batch()

if __name__ == "__main__":
	main()

