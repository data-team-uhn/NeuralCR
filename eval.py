import argparse
import random
from onto import Ontology
import numpy as np
import os
import json
from scipy import stats

def normalize(ont, hpid_filename, column=0):
    concepts = [c.strip().split()[column].replace("_",":") for c in open(hpid_filename).readlines() if c.strip()!=""]
    filtered = [ont.real_id[c] for c in concepts if c in ont.real_id] 
    # and x.replace("_",":").strip()!="HP:0003220" and x.replace("_",":").strip()!="HP:0001263"and x.replace("_",":").strip()!="HP:0001999"]
    #raw = [ont.real_id[x.replace("_",":").strip()] for x in open(hpid_filename).readlines()]
    return set([c for c in filtered if c in ont.concepts])

def get_all_ancestors(ont, hit_list):
    return set([ont.concepts[x] for hit in hit_list for x in ont.ancestor_weight[ont.concept2id[hit]]])

def get_tp_fp(positives, real_positives):
    tp = len(positives & real_positives)
    fp = len(positives) - tp
    return tp, fp

def get_fmeasure(precision, recall):
    return 2.0*precision*recall/(precision+recall) if (precision+recall)!=0 else 0.0

def get_micro_stats(matrix):
    tp = matrix['tp']
    fp = matrix['fp']
    rp = matrix['rp']
    if np.sum(tp)+np.sum(fp) == 0:
        precision = 1.0
    else:
        precision = np.sum(tp)/(np.sum(tp)+np.sum(fp))
    if np.sum(rp) == 0:
        recall = 1.0
    else:
        recall = np.sum(tp)/np.sum(rp)
    return {"precision":precision, "recall":recall,
            "fmeasure":get_fmeasure(precision, recall)}

def get_macro_stats(matrix):
    tp = matrix['tp']
    fp = matrix['fp']
    rp = matrix['rp']
    precision = np.mean(np.where(tp+fp>0, tp/(tp+fp), 1.0))
    #precision = np.mean(np.where(tp+fp>0, tp/(tp+fp), 0.0))
    recall = np.mean(np.where(rp>0, tp/rp, 1.0))
    return {"precision":precision, "recall":recall,
            "fmeasure":get_fmeasure(precision, recall)}
    
def get_extended_stats(matrix):
    tp = matrix['tp']
    fp = matrix['fp']
    rp = matrix['rp']
    tp_precision = matrix['tp_ont_precision']
    tp_recall = matrix['tp_ont_recall']

    precision = np.mean(np.where(tp+fp>0, tp_precision/(tp+fp), 1.0))
    recall = np.mean(np.where(rp>0, tp_recall/rp, 1.0))
    return {"precision":precision, "recall":recall,
            "fmeasure":get_fmeasure(precision, recall)}


def print_results(results, is_mimic=False):
    res_print = []
    styles = ["micro", "macro"]
    if not is_mimic:
        styles.append("ont")
    for style in styles: 
        for acc_type in ["precision", "recall", "fmeasure"]: 
            res_print.append(results[style][acc_type])
    if not is_mimic:
        res_print.append(results['jaccard'])

    res_print = [x*100 for x in res_print]
    if is_mimic:
        print("%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f" % tuple(res_print))
    else:
        print("%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f" % tuple(res_print))

def get_confusion_matrix_mimic(label_dir, output_dir, file_list, ont, snomed2icd, column=0):
    true_positives = []
    false_positives = []
    real_positives = []
    total_calls = 0

    for filename in file_list:
        filename = filename.strip()

        file_real_positives = set([x.strip() for x in open(label_dir+"/"+filename).readlines() if x.strip() in snomed2icd.values()])

        file_positives = normalize(ont, output_dir+"/"+filename, column)
        file_positives = set([snomed2icd[x] for x in file_positives if x in snomed2icd])

        total_calls += len(file_positives)

        tp, fp = get_tp_fp(file_positives, file_real_positives)
        true_positives.append(tp)
        false_positives.append(fp)
        real_positives.append(len(file_real_positives))

    tp = np.array(true_positives)
    fp = np.array(false_positives)
    rp = np.array(real_positives)

    matrix = {
            'tp':np.array(true_positives),
            'fp':np.array(false_positives),
            'rp':np.array(real_positives),
            'total_calls': total_calls
            }
    return matrix

def get_confusion_matrix(label_dir, output_dir, file_list, ont, column=0):
    true_positives = []
    false_positives = []
    real_positives = []

    tp_ont_recall_list = []
    tp_ont_precision_list = []

    jaccard = []

    total_calls = 0

    for filename in file_list:
        filename = filename.strip()

        file_real_positives = normalize(ont, label_dir+"/"+filename)
        file_real_positives_ont = get_all_ancestors(ont, file_real_positives)

        file_positives = normalize(ont, output_dir+"/"+filename, column)
        total_calls += len(file_positives)
        file_positives_ont = get_all_ancestors(ont, file_positives)

        tp, fp = get_tp_fp(file_positives, file_real_positives)
        true_positives.append(tp)
        false_positives.append(fp)
        real_positives.append(len(file_real_positives))

        tp_ont_recall, _ = get_tp_fp(file_positives_ont, file_real_positives)
        tp_ont_precision, _ = get_tp_fp(file_positives, file_real_positives_ont)
        tp_ont_recall_list.append(tp_ont_recall)
        tp_ont_precision_list.append(tp_ont_precision)

        if len(file_real_positives)==0:
            jaccard.append(1.0)
        else:
            jaccard.append(
                    len(file_real_positives_ont & file_positives_ont)/
                    len(file_real_positives_ont | file_positives_ont))

    tp = np.array(true_positives)
    fp = np.array(false_positives)
    rp = np.array(real_positives)

    matrix = {
            'tp':np.array(true_positives),
            'fp':np.array(false_positives),
            'rp':np.array(real_positives),
            'tp_ont_precision':np.array(tp_ont_precision_list),
            'tp_ont_recall':np.array(tp_ont_recall_list),
            'jaccard': np.mean(jaccard),
            'total_calls': total_calls
            }
    return matrix

def eval(label_dir, output_dir, file_list, ont, column=0):
    matrix = get_confusion_matrix(label_dir, output_dir, file_list, ont, column)
    ret = { "ont": get_extended_stats(matrix),
            "macro": get_macro_stats(matrix),
            "micro": get_micro_stats(matrix),
            "jaccard":matrix['jaccard']}
    return ret

def eval_mimic(label_dir, output_dir, file_list, ont, snomed2icd, column=0):
    matrix = get_confusion_matrix_mimic(label_dir, output_dir, file_list, ont, snomed2icd, column)
    ret = { "macro": get_macro_stats(matrix),
            "micro": get_micro_stats(matrix)}
    return ret

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('label_dir', help="Path to the directory where the input text files are located")
    parser.add_argument('output_dir', help="Path to the directory where the output files will be stored")
    parser.add_argument('--obofile', help="address to the ontology .obo file")
    parser.add_argument('--snomed2icd', help="address to the ontology .obo file")
    parser.add_argument('--oboroot', help="the concept in the ontology to be used as root (only this concept and its descendants will be used)")
    parser.add_argument('--file_list', help="Path to the directory where the output files will be stored")
    parser.add_argument('--comp_dir', help="Path to the directory where the output files will be stored")
    parser.add_argument('--no_error', action="store_true")
    parser.add_argument('--eval_mimic', action="store_true")
    parser.add_argument('--output_column', type=int, help="", default=0)
    args = parser.parse_args()
    if args.no_error:
        np.seterr(divide='ignore', invalid='ignore')
    

    if args.snomed2icd != None:
        with open(args.snomed2icd, 'r') as fp:
            snomed2icd = json.load(fp)

    file_list = os.listdir(args.label_dir)
    if args.file_list != None:
        file_list = [x.strip() for x in open(args.file_list).readlines()]

    ont = Ontology(args.obofile, args.oboroot)

    if args.eval_mimic:
        results = eval_mimic(args.label_dir, args.output_dir, file_list, ont, snomed2icd, column=args.output_column)
    else:
        results = eval(args.label_dir, args.output_dir, file_list, ont, column=args.output_column)
    print_results(results, args.eval_mimic)

if __name__ == "__main__":
	main()
