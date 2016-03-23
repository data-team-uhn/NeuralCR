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

def randomize_triplet(t):
	new_t = [x for x in t] #.append(0)
	new_t.append(np.array([1.0, 0.0]))
	if random.random() > 0.5:
		new_t[1], new_t[2] = new_t[2], new_t[1]
		new_t[3] = np.array([0.0, 1.0])
	return new_t

def tokenize(phrase):
	return phrase.lower().replace(',',' ').replace('-',' ').replace(';', ' ').strip().split()	

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

def read_oboFile(oboFile):
	concepts={}
	neighbour={}
	while True:
		line=oboFile.readline()
		if line == "":
			break
		tokens=line.strip().split(" ")
		if tokens[0]=="id:":
			hp_id=tokens[1]
			concepts[hp_id] = {}
			neighbour[hp_id] = []
			concepts[hp_id]['kids'] = []

		if tokens[0] == "name:":
			concepts[hp_id]['names'] = [' '.join(tokens[1:])]
		if tokens[0] == "synonym:":
			last_index = (i for i,v in enumerate(tokens) if v.endswith("\"")).next()
			concepts[hp_id]['names'].append( ' '.join(tokens[1:last_index+ 1]).strip("\"") )

	oboFile.seek(0)
	while True:
		line=oboFile.readline()
		if line == "":
			break
		tokens=line.strip().split(" ")
		if tokens[0]=="id:":
			hp_id=tokens[1]

		if tokens[0]=="is_a:":
			concepts[tokens[1]]['kids'].append(hp_id)
			neighbour[tokens[1]].append(hp_id)
			neighbour[hp_id].append(tokens[1])

	return concepts, neighbour

def generate_triplets_graph_structure(concept_ids, concepts, neighbour, relevant_interval, irrelevant_interval):
	relevant_concepts = {}
	irrelevant_concepts = {}

	for v in concept_ids:
		relevant_concepts[v] = [u for u in bfs(neighbour, v, relevant_interval[0], relevant_interval[1])]# if u in concept_ids]
		irrelevant_concepts[v] = [u for u in bfs(neighbour, v, irrelevant_interval[0], irrelevant_interval[1])]# if u in concept_ids] #bfs(neighbour, v, 2, 2)
		shuffle(relevant_concepts[v])
		shuffle(irrelevant_concepts[v])

	triplets = []

	for v in concept_ids:
		if len(neighbour[v]) == 0:
			continue
		loc_triplets = []
		pairs = [pair for pair in product(relevant_concepts[v], irrelevant_concepts[v])]
		for rc, ic in random.sample(pairs, min(len(pairs),len(relevant_concepts[v]))):
			v_name, rc_name, ic_name = [ random.choice(concepts[concept]['names']) for concept in [v, rc, ic] ] #in product(concepts[v]['names'], concepts[rc]['names'], concepts[ic]['names']):
			loc_triplets.append([v_name, rc_name, ic_name])
		triplets += loc_triplets
	return triplets

def generate_triplets_synonyms(concept_ids, concepts, neighbour, irrelevant_interval):

	irrelevant_concepts = {}
	for v in concept_ids:
		irrelevant_concepts[v] = [u for u in bfs(neighbour, v, irrelevant_interval[0], irrelevant_interval[1])]# if u in concept_ids] #bfs(neighbour, v, 2, 2)
		shuffle(irrelevant_concepts[v])

	triplets = []
	ct=0
	for v in concept_ids:
		for x_name1 in concepts[v]['names']:
			for x_name2 in concepts[v]['names']:
				if x_name1 == x_name2:
					continue
				for t in range(min(3,len(irrelevant_concepts[v]))):
					irrelevant_concept = irrelevant_concepts[v][t]
					triplets.append([x_name1, x_name2, random.choice(concepts[irrelevant_concept]['names'])])
	return triplets

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



def main():
	oboFile=open("hp.obo")
	vectorFile=open("test_vectors.txt")

	concepts, neighbour = read_oboFile(oboFile)
	wordVector={}
	for line in vectorFile:
		tokens = line.strip().split(" ")
		wordVector[tokens[0]] = np.array(map(float,tokens[1:]))
		
	concept_ids = [v for v in concepts.keys()]
	shuffle(concept_ids)
	train_set = set(concept_ids)
	train_set = set(concept_ids[:int(len(concept_ids)*0.8)])
	validation_set = set(concept_ids[int(len(concept_ids)*0.8):int(len(concept_ids)*0.9)])
	test_set = set(concept_ids[int(len(concept_ids)*0.9):])

#	train_triplets = generate_triplets_graph_structure(train_set, concepts, neighbour, [1,1], [3,5]) + generate_triplets_graph_structure(train_set, concepts, neighbour, [1,1], [2,2]) + generate_triplets_synonyms(train_set, concepts, neighbour, [3,5])

	word_limit=10
	data={
			'training' : postprocess_triplets( generate_triplets_graph_structure(train_set, concepts, neighbour, [1,1], [3,5]) + generate_triplets_graph_structure(train_set, concepts, neighbour, [1,1], [2,2]) + generate_triplets_synonyms(train_set, concepts, neighbour, [3,5])  + generate_triplets_synonyms(train_set, concepts, neighbour, [2,2]) + generate_triplets_synonyms(train_set, concepts, neighbour, [1,1]), wordVector, word_limit ),

			'validation_graph_3_5' : postprocess_triplets( generate_triplets_graph_structure(validation_set, concepts, neighbour, [1,1], [3,5]) , wordVector, word_limit ),
			'validation_graph_2_2' : postprocess_triplets( generate_triplets_graph_structure(validation_set, concepts, neighbour, [1,1], [2,2]) , wordVector, word_limit ),

			'validation_synonym_3_5' : postprocess_triplets( generate_triplets_synonyms(validation_set, concepts, neighbour, [3,5]) , wordVector, word_limit ),
			'validation_synonym_2_2' : postprocess_triplets( generate_triplets_synonyms(validation_set, concepts, neighbour, [2,2]) , wordVector, word_limit ),
			'validation_synonym_1_1' : postprocess_triplets( generate_triplets_synonyms(validation_set, concepts, neighbour, [1,1]) , wordVector, word_limit ),

			'test_graph_3_5' : postprocess_triplets( generate_triplets_graph_structure(test_set, concepts, neighbour, [1,1], [3,5]) , wordVector, word_limit ),
			'test_graph_2_2' : postprocess_triplets( generate_triplets_graph_structure(test_set, concepts, neighbour, [1,1], [2,2]) , wordVector, word_limit ),

			'test_synonym_3_5' : postprocess_triplets( generate_triplets_synonyms(test_set, concepts, neighbour, [3,5]) , wordVector, word_limit ),
			'test_synonym_2_2' : postprocess_triplets( generate_triplets_synonyms(test_set, concepts, neighbour, [2,2]) , wordVector, word_limit ),
			'test_synonym_1_1' : postprocess_triplets( generate_triplets_synonyms(test_set, concepts, neighbour, [1,1]) , wordVector, word_limit ),
			}

	store_data(data, "./data_files/")

if __name__ == "__main__":
	main()

