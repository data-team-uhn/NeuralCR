import argparse
import json
import random
from onto import Ontology
import numpy as np
import os
from scipy import stats
from eval import normalize, get_all_ancestors, get_tp_fp, get_confusion_matrix, get_fmeasure


def prob_compare_dirich(label_dir, output_dir_a, output_dir_b, file_list, ont, column_a=0, column_b=0):    
    n_a_true_b_false = 0
    n_b_true_a_false = 0

    for filename in file_list:
        filename = filename.strip()
        file_real_positives = normalize(ont, label_dir+"/"+filename)
        file_real_positives = get_all_ancestors(ont, file_real_positives)

        file_positives_a = normalize(ont, output_dir_a+"/"+filename, column_a)
        file_positives_a = get_all_ancestors(ont, file_positives_a)

        file_positives_b = normalize(ont, output_dir_b+"/"+filename, column_b)
        file_positives_b = get_all_ancestors(ont, file_positives_b)

        n_a_true_b_false += len(
                (file_positives_a - file_positives_b) & file_real_positives)
        n_a_true_b_false += len(
                (file_positives_b - file_positives_a) - file_real_positives)

        n_b_true_a_false += len(
                (file_positives_b - file_positives_a) & file_real_positives)
        n_b_true_a_false += len(
                (file_positives_a - file_positives_b) - file_real_positives)

    samples = np.random.dirichlet(
            (n_a_true_b_false+1, n_b_true_a_false+1), 10000)
    sig = np.mean(samples[:,0]>samples[:,1])


    return {'n_a_true_b_false': n_a_true_b_false, 'n_b_true_a_false': n_b_true_a_false, 'sig': sig}

def get_matrix_dict_mimic(label_dir, output_dir, file_list, ont, snomed2icd, column=0):
    true_positives = {}
    false_positives = {}
    real_positives = {}

    total_calls = 0

    for filename in file_list:
        filename = filename.strip()

        file_real_positives = set([x.strip() for x in open(label_dir+"/"+filename).readlines() if x.strip() in snomed2icd.values()])

        file_positives = normalize(ont, output_dir+"/"+filename, column)
        file_positives = set([snomed2icd[x] for x in file_positives if x in snomed2icd])

        total_calls += len(file_positives)

        tp, fp = get_tp_fp(file_positives, file_real_positives)
        true_positives[filename] = tp
        false_positives[filename] = fp
        real_positives[filename] = len(file_real_positives)

    tp = np.array(true_positives)
    fp = np.array(false_positives)
    rp = np.array(real_positives)

    matrix = {
            'ord':{
                'tp':true_positives,
                'fp':false_positives,
                'rp':real_positives},
            'total_calls': total_calls
            }
    return matrix


def get_matrix_dict(label_dir, output_dir, file_list, ont, column=0):
    true_positives = {}
    false_positives = {}
    real_positives = {}

    true_positives_ont = {}
    false_positives_ont = {}
    real_positives_ont = {}

    jaccard = {}

    total_calls = 0

    for filename in file_list:
        filename = filename.strip()

        file_real_positives = normalize(ont, label_dir+"/"+filename)
        file_real_positives_ont = get_all_ancestors(ont, file_real_positives)

        file_positives = normalize(ont, output_dir+"/"+filename, column)
        total_calls += len(file_positives)
        file_positives_ont = get_all_ancestors(ont, file_positives)

        tp, fp = get_tp_fp(file_positives, file_real_positives)
        true_positives[filename] = tp
        false_positives[filename] = fp
        real_positives[filename] = len(file_real_positives)

        tp, fp = get_tp_fp(file_positives_ont, file_real_positives_ont)
        true_positives_ont[filename] = tp
        false_positives_ont[filename] = fp
        real_positives_ont[filename] = len(file_real_positives_ont)

        if len(file_real_positives)==0:
            jaccard[filename] = 1.0
        else:
            jaccard[filename] = (
                    len(file_real_positives_ont & file_positives_ont)/
                    len(file_real_positives_ont | file_positives_ont))

    tp = np.array(true_positives)
    fp = np.array(false_positives)
    rp = np.array(real_positives)

    tp_ont = np.array(true_positives_ont)
    fp_ont = np.array(false_positives_ont)
    rp_ont = np.array(real_positives_ont)

    matrix = {
            'ord':{
                'tp':true_positives,
                'fp':false_positives,
                'rp':real_positives},
            'ont':{
                'tp':true_positives_ont,
                'fp':false_positives_ont,
                'rp':real_positives_ont},
            'jaccard': jaccard,
            'total_calls': total_calls
            }
    return matrix

def paired_test_per_document_fscore(label_dir, output_dir_a, output_dir_b, file_list, ont, column_a, column_b, is_mimic=False, snomed2icd=None):
    dicts = {}
    if not is_mimic:
        dicts['model_a'] = get_matrix_dict(label_dir, output_dir_a, file_list, ont, column=column_a)
        dicts['model_b'] = get_matrix_dict(label_dir, output_dir_b, file_list, ont, column=column_b)
    else:
        dicts['model_a'] = get_matrix_dict_mimic(label_dir, output_dir_a, file_list, ont, snomed2icd, column=column_a)
        dicts['model_b'] = get_matrix_dict_mimic(label_dir, output_dir_b, file_list, ont, snomed2icd, column=column_b)
    heading = [('ord', 'tp'),('ord', 'fp'),('ord', 'rp')]
    f1_list = {'model_a':[],'model_b':[]}
    for filename in sorted(file_list):
        for mod in sorted(dicts.keys()):
            tp = dicts[mod]['ord']['tp'][filename]
            fp = dicts[mod]['ord']['fp'][filename]
            rp = dicts[mod]['ord']['rp'][filename]
            recall = tp/rp if rp!=0 else 1.0
            precision = tp/(tp+fp) if tp+fp!=0 else 1.0
            f1 = get_fmeasure(precision, recall)
            f1_list[mod].append(f1)

    return {'mean_a': np.mean(f1_list['model_a']),
            'mean_b': np.mean(f1_list['model_b']),
            't-test': stats.ttest_rel(
                np.array(f1_list['model_a']), np.array(f1_list['model_b'])),
            'wilcoxon': stats.wilcoxon(
                np.array(f1_list['model_a']), np.array(f1_list['model_b'])),
            'wilcoxon-pratt': stats.wilcoxon(
                np.array(f1_list['model_a']), np.array(f1_list['model_b']),
                zero_method='pratt')}

def prob_compare_fmeasure(matrix_a, matrix_b):
    lambda_parameter = 0.5
    shape_parameter = 1.0
    sample_size = 100000
    u_a = np.random.gamma(
            np.sum(matrix_a['tp']) + lambda_parameter, 2*shape_parameter, [sample_size])
    v_a = np.random.gamma(
            np.sum(matrix_a['fp']) + np.sum(matrix_a['rp']) - np.sum(matrix_a['tp']) + 2*lambda_parameter,
            shape_parameter, [sample_size])
    f_a = u_a/(u_a + v_a)


    u_b = np.random.gamma(
            np.sum(matrix_b['tp']) + lambda_parameter, 2*shape_parameter, [sample_size])
    v_b = np.random.gamma(
            np.sum(matrix_b['fp']) + np.sum(matrix_b['rp']) - np.sum(matrix_b['tp']) + 2*lambda_parameter,
            shape_parameter, [sample_size])
    f_b = u_b/(u_b + v_b)
    return {'f_a': np.mean(f_a),'f_b': np.mean(f_b), 'sig': np.mean(f_a>f_b)}


def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('label_dir', help="Path to the directory where the input text files are located")
    parser.add_argument('output_dir_a', help="Path to the directory where the output files will be stored")
    parser.add_argument('output_dir_b', help="Path to the directory where the output files will be stored")
    parser.add_argument('--obofile', help="address to the ontology .obo file")
    parser.add_argument('--oboroot', help="the concept in the ontology to be used as root (only this concept and its descendants will be used)")

    parser.add_argument('--file_list', help="Path to the directory where the output files will be stored")
    parser.add_argument('--comp_dir', help="Path to the directory where the output files will be stored")
    parser.add_argument('--output_column_a', type=int, help="", default=0)
    parser.add_argument('--output_column_b', type=int, help="", default=0)
    parser.add_argument('--eval_mimic', action="store_true")
    parser.add_argument('--snomed2icd', help="address to the ontology .obo file")
    args = parser.parse_args()

    file_list = os.listdir(args.label_dir)
    if args.file_list != None:
        file_list = [x.strip() for x in open(args.file_list).readlines()]
    random.shuffle(file_list)

    ont = Ontology(args.obofile, args.oboroot)

    if args.snomed2icd != None:
        with open(args.snomed2icd, 'r') as fp:
            snomed2icd = json.load(fp)

    if not args.eval_mimic:
        res = paired_test_per_document_fscore(args.label_dir, args.output_dir_a, args.output_dir_b, file_list, ont, args.output_column_a, args.output_column_b)
    else:
        res = paired_test_per_document_fscore(args.label_dir, args.output_dir_a, args.output_dir_b, file_list, ont, args.output_column_a, args.output_column_b, True, snomed2icd)
    for cat in sorted(res.keys()):
        if cat.startswith('mean'):
            print(cat, ' :: ', "{:.4f}".format(res[cat]))
        else:
            print(cat,"p-value= {:.4f}".format(res[cat].pvalue))

    '''
    matrix_a = get_confusion_matrix(args.label_dir, args.output_dir_a, file_list, ont, column=args.output_column_a)
    matrix_b = get_confusion_matrix(args.label_dir, args.output_dir_b, file_list, ont, column=args.output_column_b)
    res = prob_compare_fmeasure(matrix_a['ord'], matrix_b['ord'])
    for cat in ['f_a', 'f_b']:
        print(cat, ' :: ', "{:.4f}".format(res[cat]))
    print('sig', ' :: ', "{:.4f}".format(res['sig']), '(p-value=', "{:.4f}".format(1-res['sig']), ')')
    '''

if __name__ == "__main__":
	main()
