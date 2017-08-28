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
from onto import Ontology

def normalize(ont, hpid_filename):
    raw = [ont.real_id[x.replace("_",":").strip()] for x in open(hpid_filename).readlines() if x.replace("_",":").strip() in ont.real_id]
    #raw = [ont.real_id[x.replace("_",":").strip()] for x in open(hpid_filename).readlines()]
    return set([x for x in raw if x in ont.concepts])

def get_all_ancestors(ont, hit_list):
    return set([ont.concepts[x] for hit in hit_list for x in ont.ancestrs[ont.concept2id[hit]]])

def eval(label_dir, output_dir, file_list, ont, comp_dir=None):
    total_precision = 0
    total_recall = 0
    total_docs = 0

    total_relevant = 0
    total_positives = 0
    total_true_pos = 0

    jaccard_sum = 0
    false_pos_all = {}

    for filename in file_list:
    #for filename in open(file_list).readlines():
        filename = filename.strip()
        real = normalize(ont, label_dir+"/"+filename)
        extended_real = get_all_ancestors(ont, real)
        if comp_dir!=None:
            comp_positives = normalize(ont, comp_dir+"/"+filename)
            extended_comp_positives = get_all_ancestors(ont, comp_positives)
        positives = normalize(ont, output_dir+"/"+filename)
        extended_positives = get_all_ancestors(ont, positives)
        true_pos = [x for x in positives if x in real]
        false_pos = [x for x in positives if x not in real]
        for x in false_pos:
            if x not in false_pos_all:
                false_pos_all[x] = 0
            false_pos_all[x] += 1


        precision = 0
        if len(positives)!=0:
            precision = 1.0*len(true_pos)/len(positives)

        recall = 0
        if len(real)!=0:
            recall = 1.0*len(true_pos)/len(real)

        total_docs += 1
        total_precision += precision
        total_recall += recall
        #print filename, '\t', precision, '\t', recall

        total_relevant += len(real)
        total_positives += len(positives)
        total_true_pos += len(true_pos)
        
        if len(extended_real | extended_positives) == 0:
            jaccard = 1.0
        else:
            jaccard = 1.0 * len(extended_real & extended_positives) / len(extended_real | extended_positives)
            if comp_dir!=None and len(extended_real | extended_comp_positives)!=0:
                jaccard_comp = 1.0 * len(extended_real & extended_comp_positives) / len(extended_real | extended_comp_positives)
                if jaccard<jaccard_comp:
                    print filename

        jaccard_sum += jaccard

    precision = total_precision/total_docs
    recall = total_recall/total_docs
    fmeasure = 2.0*precision*recall/(precision+recall)

    if total_positives>0:
        mprecision = 1.0*total_true_pos/total_positives
    else:
        mprecision = 1.0
    mrecall = 1.0*total_true_pos/total_relevant
    mfmeasure = 2.0*mprecision*mrecall/(mprecision+mrecall)

    jaccard_mean = jaccard_sum/total_docs

    for hp,ct in sorted(false_pos_all.iteritems(), key=lambda (k,v): (-v,k)):
        print hp, ont.names[hp][0], ct
    ret = {"vanila":{"precision":precision, "recall":recall, "fmeasure":fmeasure}, "micro":{"precision":mprecision, "recall":mrecall, "fmeasure":mfmeasure}, "jaccard":jaccard_mean}
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
    parser.add_argument('--comp_dir', help="Path to the directory where the output files will be stored")
    parser.add_argument('--repdir', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="checkpoints/")
    args = parser.parse_args()

    ont = Ontology('data/hp.obo',"HP:0000118")
    results = eval(args.label_dir, args.output_dir, open(args.file_list).readlines(), ont, args.comp_dir)
    res_print = []
    for style in ["micro", "vanila"]: 
        for acc_type in ["precision", "recall", "fmeasure"]: 
            res_print.append(results[style][acc_type])
    res_print.append(results['jaccard'])

    res_print = [x*100 for x in res_print]
    print "%.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f\\\\" % tuple(res_print)
    #print "%.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f\\\\" % tuple(res_print)

    #roc(args.label_dir, args.output_dir, args.file_list, rd)
if __name__ == "__main__":
	main()
