import argparse
import ncrmodel 
import numpy as np
import os
from onto import Ontology
import json
import fasttext
import pickle 
import tensorflow as tf

def create_negatives(text, num):
    neg_tokens = ncrmodel.tokenize(text)
    indecies = np.random.choice(len(neg_tokens), num)
    lengths = np.random.randint(1, 10, num)
    negative_phrases = [' '.join(neg_tokens[indecies[i]:indecies[i]+lengths[i]])
                                for i in range(num)]
    return negative_phrases

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--obofile', help="address to the ontology .obo file")
    parser.add_argument('--oboroot', help="the concept in the ontology to be used as root (only this concept and its descendants will be used)")
    parser.add_argument('--fasttext', help="address to the fasttext word vector file")
    parser.add_argument('--neg_file', help="address to the negative corpus", default="")
    parser.add_argument('--output', help="address to the directroy where the trained model will be stored")

    parser.add_argument('--flat', action="store_true")
    parser.add_argument('--cl1', type=int, help="cl1", default=1024)
    parser.add_argument('--cl2', type=int, help="cl2", default=1024)
    parser.add_argument('--lr', type=float, help="lr", default=1/512)
    parser.add_argument('--batch_size', type=int, help="batch_size", default=256)
    parser.add_argument('--max_sequence_length', type=int, help="max_sequence_length", default=50)
    parser.add_argument('--epochs', type=int, help="epochs", default=50)
    args = parser.parse_args()

    word_model = fasttext.load_model(args.fasttext)
    ont = Ontology(args.obofile,args.oboroot)

    model = ncrmodel.NCRModel(args, ont, word_model)
    if args.neg_file == "":
        model.init_training()
    else:
        wiki_file = open(args.neg_file, errors='replace')
        wiki_text = wiki_file.read()
        wiki_negs = set(create_negatives(wiki_text[:10000000], 10000))
        model.init_training(wiki_negs)


    for epoch in range(args.epochs):
        print("Epoch :: "+str(epoch))
        model.train_epoch(verbose=True)

    param_dir = args.output
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    model.save_params(param_dir)
    
    pickle.dump(ont, open(param_dir+'/ont.pickle',"wb" )) 

    with open(param_dir+'/config.json', 'w') as fp:
        json.dump(vars(args),fp)

if __name__ == "__main__":
    main()
