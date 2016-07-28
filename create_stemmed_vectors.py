from random import shuffle
import random
import numpy as np
import cPickle as pickle
import nltk
from itertools import product

def main():
	vectorFile=open("vectors.txt")
	stemmed_word_vectors = {}
	stemmed_word_counts = {}
	words = []

	for i,line in enumerate(vectorFile):
		tokens = line.strip().split(" ")
		stemmed = nltk.stem.PorterStemmer().stem(tokens[0])
		#print tokens[0], stemmed
		if stemmed not in stemmed_word_vectors:
			stemmed_word_vectors[stemmed] = np.array(map(float,tokens[1:]))
			stemmed_word_counts[stemmed] = 1
			words.append(stemmed)
		else:
			stemmed_word_vectors[stemmed] += np.array(map(float,tokens[1:]))
			stemmed_word_counts[stemmed] += 1

	for word in words:
		print word,
		for s in (stemmed_word_vectors[word]/stemmed_word_counts[word]):
			print s,
		print ""


if __name__ == "__main__":
	main()

