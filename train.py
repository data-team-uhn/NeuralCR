import argparse
import ncrmodel 
import numpy as np
import os
from onto import Ontology
import json
import fasttext
import cPickle as pickle 

def create_negatives(text, num):
    neg_tokens = ncrmodel.tokenize(text)
    indecies = np.random.choice(len(neg_tokens), num)
    lengths = np.random.randint(1, 10, num)
    negative_phrases = [' '.join(neg_tokens[indecies[i]:indecies[i]+lengths[i]])
                                for i in range(num)]
    return negative_phrases

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--obofile', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--oboroot', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--fasttext', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--neg_file', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--output', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--no_negs', help="Would not include negative samples during training", action="store_true")
    parser.add_argument('--no_agg', action="store_true")
    args = parser.parse_args()
    print args

    word_model = fasttext.load_model(args.fasttext)
    ont = Ontology(args.obofile,args.oboroot)

    config = ncrmodel.Config
    config.agg = not args.no_agg

    model = ncrmodel.NCRModel(config, ont, word_model)
    if args.no_negs:
        model.init_training()
    else:
        wiki_text = open(args.neg_file).read()
        wiki_negs = set(create_negatives(wiki_text[:10000000], 10000))
        model.init_training(wiki_negs)


    num_epochs = 50
    for epoch in range(num_epochs):
        print "Epoch ::", epoch
        model.train_epoch(verbose=False)

    param_dir = args.output
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    model.save_params(param_dir)
    
    pickle.dump(ont, open(param_dir+'/ont.pickle',"wb" )) 

    with open(param_dir+'/config.json', 'w') as fp:
        json.dump(config.__dict__,fp)

if __name__ == "__main__":
    main()
