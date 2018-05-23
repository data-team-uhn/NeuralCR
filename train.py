import argparse
import ncrmodel 
import numpy as np
import os
from onto import Ontology
import json
import fasttext
import pickle 
import tensorflow as tf
import accuracy

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

    parser.add_argument('--phrase_val', help="address to the file containing labeled phrases for validation")

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

    if args.phrase_val != None: 
        samples = accuracy.prepare_phrase_samples(model.ont, args.phrase_val, True)

    for epoch in range(args.epochs):
        print("Epoch :: "+str(epoch))
        model.train_epoch(verbose=True)
        if args.phrase_val != None and epoch%5==0 and epoch>0: 
            res = model.get_match(list(samples.keys()), 1)
            missed = [x for i,x in enumerate(samples) if samples[x] not in [r[0] for r in res[i]]]
            print("R@1: "+ str((len(samples)-len(missed))/len(samples)))

            res = model.get_match(list(samples.keys()), 5)
            missed = [x for i,x in enumerate(samples) if samples[x] not in [r[0] for r in res[i]]]
            print("R@5: "+ str((len(samples)-len(missed))/len(samples)))
        if epoch%5==0 and epoch>0: 
            #for x in model.get_match('retina cancer', 5):
            for x in model.get_match('blood examination', 5):
                print(x[0], (ont.names[x[0]] if x[0]!='None' else x[0]), x[1])
#            print(model.get_match('retina cancer', 5))

    param_dir = args.output
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    model.save_params(param_dir)
    
    pickle.dump(ont, open(param_dir+'/ont.pickle',"wb" )) 

    with open(param_dir+'/config.json', 'w') as fp:
        json.dump(vars(args),fp)

if __name__ == "__main__":
    main()
