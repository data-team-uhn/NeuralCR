import argparse
import phrase_model 
import accuracy
import fasttext_reader as reader
import numpy as np
import sys
import sent_accuracy
import time
import os
import h5py
from onto import Ontology
import fasttext
from eval import eval
from train_phrase import create_output_dir, print_res

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--exp_name', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="newExpForOmim")
    parser.add_argument('--no_negs', help="Would not include negative samples during training", action="store_true")
    parser.add_argument('--no_agg', action="store_true")
    args = parser.parse_args()


    param_str = ""
    if args.no_negs:
        param_str+='_nonegs'
    else:
        param_str+='_negs'
    if args.no_agg:
        param_str+='_noagg'
    else:
        param_str+='_agg'

    omim_data = '../../datasets/omim/'
    name = args.exp_name+param_str
    params_dir = 'params_experiment'+param_str
    output_dir_prefix = "outputs_" + name + "_"

    thetas = {'_negs_agg':0.85, '_nonegs_agg':0.8, '_negs_noagg':0.8, '_nonegs_noagg':0.75}
    best_theta = thetas[param_str]

    config = phrase_model.Config
    config.agg = not args.no_agg

    ont = Ontology('data/hp.obo',"HP:0000118")
    word_model = fasttext.load_model('data/model_pmc.bin')
    model = phrase_model.NCRModel(config, ont, word_model)
    model.load_params(params_dir)

    omim_output_dir = output_dir_prefix+'omimbig_'+str(best_theta)+"/"
    create_output_dir(model, best_theta, omim_data+"/text/", omim_output_dir)
    omim_results = eval(omim_data+"labels/", omim_output_dir, os.listdir(omim_data+"/text/"), model.ont)

    print "OMIM test results:"
    outfile = open("report_"+name+".txt","a")
    outfile.write("OMIM test results:\n")
    print print_res(omim_results)
    outfile.write(print_res(omim_results)+"\n")


if __name__ == "__main__":
    main()
