from fasttext_reader import tokenize
import numpy as np
import cPickle as pickle
import sys

def create_negatives(text, num):
    neg_tokens = tokenize(text)
    print len(neg_tokens)

    indecies = np.random.choice(len(neg_tokens), num)
    lengths = np.random.randint(1, 10, num)

    negative_phrases = [' '.join(neg_tokens[indecies[i]:indecies[i]+lengths[i]])
                                for i in range(num)]
    return negative_phrases

def main():
    '''
    text = ' '.join(pickle.load(open('data/wiki-samples.p', 'rb')))
    print text
    exit()
    '''
    text = open('wiki_text').read()
    print "H"
    negs = create_negatives(text[:10000000], 100000)
    exit()
    negs = open('neggs').read().split('\n')
    print negs[:5]
    print len(negs)
    #print '\n'.join(negs)


if __name__ == "__main__":
	main()
