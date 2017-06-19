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

def normalize(rd, hpid_filename):
    raw = [rd.real_id[x.replace("_",":").strip()] for x in open(hpid_filename).readlines()]
    return set([x for x in raw if x in rd.concepts])

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('label_dir', help="Path to the directory where the input text files are located")
    parser.add_argument('output_dir', help="Path to the directory where the output files will be stored")
    parser.add_argument('--repdir', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="checkpoints/")
    args = parser.parse_args()

    rd = reader.Reader("data/")

    total_precision = 0
    total_recall = 0
    total_docs = 0

    total_relevant = 0
    total_positives = 0
    total_true_pos = 0

    for filename in os.listdir(args.label_dir):
        relevant = normalize(rd, args.label_dir+"/"+filename)
        positives = normalize(rd, args.output_dir+"/"+filename)
        true_pos = [x for x in positives if x in relevant]

        precision = 1
        if len(positives)!=0:
            precision = 1.0*len(true_pos)/len(positives)

        recall = 1
        if len(relevant)!=0:
            recall = 1.0*len(true_pos)/len(relevant)

        total_docs += 1
        total_precision += precision
        total_recall += recall

        total_relevant += len(relevant)
        total_positives += len(positives)
        total_true_pos += len(true_pos)

    print "Precision:", total_precision/total_docs
    print "Recall:", total_recall/total_docs

    print "Micro Precision:", 1.0*total_true_pos/total_positives
    print "Micro Recall:", 1.0*total_true_pos/total_relevant

if __name__ == "__main__":
	main()
