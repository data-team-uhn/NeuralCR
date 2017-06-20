import argparse
import phraseConfig
import phrase_model 
import accuracy
import fasttext_reader as reader
import numpy as np
import sys
import sent_level
import sent_accuracy
import time
import os


def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('input_dir', help="Path to the directory where the input text files are located")
    parser.add_argument('output_dir', help="Path to the directory where the output files will be stored")
    parser.add_argument('--repdir', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="checkpoints/")
    args = parser.parse_args()

    config = phraseConfig.Config
    rd = reader.Reader("data/", config.include_negs)
    model = phrase_model.NCRModel(config, rd)
    model.load_params(args.repdir)
#    model.set_anchors()

    text_ant = sent_level.TextAnnotator(model)

    for theta in [0.7]:
    #for theta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for filename in os.listdir(args.input_dir):
            print filename
            text = open(args.input_dir+"/"+filename).read()
            predictions = text_ant.process_text(text, theta)
            with open(args.output_dir+"_"+str(theta)+"/"+filename,"w") as fw:
                for y in predictions:
                    fw.write(y[2]+"\n")

if __name__ == "__main__":
	main()
