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

def eval(label_dir, output_dir, file_list, rd):
    total_precision = 0
    total_recall = 0
    total_docs = 0

    total_relevant = 0
    total_positives = 0
    total_true_pos = 0

    for filename in open(file_list).readlines():
        filename = filename.strip()
        relevant = normalize(rd, label_dir+"/"+filename)
        positives = normalize(rd, output_dir+"/"+filename)
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

    precision = total_precision/total_docs
    recall = total_recall/total_docs
    fmeasure = 2.0*precision*recall/(precision+recall)

    mprecision = 1.0*total_true_pos/total_positives
    mrecall = 1.0*total_true_pos/total_relevant
    mfmeasure = 2.0*mprecision*mrecall/(mprecision+mrecall)

    ret = {"vanila":{"precision":precision, "recall":recall, "fmeasure":fmeasure}, "micro":{"precision":mprecision, "recall":mrecall, "fmeasure":mfmeasure}}
    return ret
    print "Precision:", precision
    print "Recall:", recall 
    print "F-measure:", 2.0*precision*recall/(precision+recall)

    print "Micro Precision:", mprecision 
    print "Micro Recall:", mrecall 
    print "Micro F-measure:", 2.0*mprecision*mrecall/(mprecision+mrecall)

def roc(label_dir, output_dir, file_list, rd):
    exps = []
    for i in range(1,10):
        exps.append(eval(label_dir, output_dir+"_0."+str(i), file_list, rd))
#    exps = sorted(exps, key= lambda x: x["micro"]["recall"])
    recalls = [x["micro"]["recall"] for x in exps]
    precisions = [x["micro"]["precision"] for x in exps]
    print recalls
    print precisions

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot([x/10.0 for x in range(1,10)], precisions, 'r', label="precision")
    plt.plot([x/10.0 for x in range(1,10)], recalls, 'b', label="recall")
    plt.ylabel('accuracy')
    plt.xlabel('Threshold')
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.legend(loc='upper left')
#    plt.show()
    plt.savefig("roc.pdf")


def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('label_dir', help="Path to the directory where the input text files are located")
    parser.add_argument('output_dir', help="Path to the directory where the output files will be stored")
    parser.add_argument('file_list', help="Path to the directory where the output files will be stored")
    parser.add_argument('--repdir', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="checkpoints/")
    args = parser.parse_args()

    rd = reader.Reader("data/")

    results = eval(args.label_dir, args.output_dir, args.file_list, rd)
    res_print = []
    for style in ["micro", "vanila"]: 
        for acc_type in ["precision", "recall", "fmeasure"]: 
            res_print.append(results[style][acc_type])
    print "%.4f & %.4f & %.4f & %.4f & %.4f & %.4f\\\\" % tuple(res_print)

    #roc(args.label_dir, args.output_dir, args.file_list, rd)
if __name__ == "__main__":
	main()
