import argparse
import phraseConfig
import phrase_model 
import accuracy
import fasttext_reader as reader
import numpy as np
import sys
import time
import os


from sent_model import SentAnt
from extractor import ExtractModel, ExtConfig
from phrase_model import NCRModel
import phraseConfig
import fasttext
from onto import Ontology


def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('input_dir', help="Path to the directory where the input text files are located")
    parser.add_argument('output_dir', help="Path to the directory where the output files will be stored")
    parser.add_argument('--repdir', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="checkpoints/")
    args = parser.parse_args()

    #######
    word_model = fasttext.load_model('data/model_pmc.bin')
    ont = Ontology('data/hp.obo',"HP:0000118")
    ncrmodel = NCRModel(phraseConfig.Config, ont, word_model)
    ncrmodel.load_params('checkpoints/')

    extractmodel = ExtractModel(ExtConfig, word_model)
    extractmodel.load_params('ext_params')

    sentant = SentAnt(ncrmodel, extractmodel)
    #######
 



    '''
    config = phraseConfig.Config
    rd = reader.Reader("data/", config.include_negs)
    model = phrase_model.NCRModel(config, rd)
    model.load_params(args.repdir)
    '''
#    model.set_anchors()


    for theta in [0.3, 0.5, 0.7]:
        for theta2 in [0.3, 0.5, 0.7, 0.9]:
            ct=0
            output_dir = args.output_dir+"/res_"+str(theta)+"_"+str(theta2)+"/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for filename in os.listdir(args.input_dir):
                print output_dir+"/"+filename
                text = open(args.input_dir+"/"+filename).read()
                predictions = sentant.process_text(text, theta, theta2)
                with open(output_dir+"/"+filename,"w") as fw:
                    for y in predictions:
                        fw.write(y[2]+"\n")
                ct += 1
    #            if ct == 15:
    #                exit()

if __name__ == "__main__":
	main()
