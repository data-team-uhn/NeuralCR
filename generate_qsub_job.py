#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def file_printline(f, line):
  f.write(line)
  f.write('\n')

def generate_json(qsub_path, job_id, training_args):
  f_json = open("{}/{}.json".format(qsub_path, job_id), 'w')
  file_printline(f_json, training_args.to_json())
  f_json.close()
  os.chmod("{}/{}.json".format(qsub_path, job_id), int('444', 8)) #Make it executable


def generate_script(qsub_path, job_id):
  f_qsub = open("{}/{}.sh".format(qsub_path, job_id), 'w')
  file_printline(f_qsub, "#!/bin/bash")
  
  file_printline(f_qsub, "") #Blank line, make it nicer to read
  
  file_printline(f_qsub, "#PBS -l mem=32gb,vmem=32gb")
  file_printline(f_qsub, "#PBS -l nodes=1:ppn=2")
  file_printline(f_qsub, "#PBS -l walltime=72:00:00")
  file_printline(f_qsub, "#PBS -j oe")
  file_printline(f_qsub, "#PBS -o $HOME/ncr_logs/{}/".format(job_id))
  
  file_printline(f_qsub, "") #Blank line, make it nicer to read
  
  file_printline(f_qsub, "module load tensorflow/1.13.1-py3410-cpu")
  file_printline(f_qsub, "pip3 install 'pybind11' --user")
  file_printline(f_qsub, "pip3 install 'fasttext==0.9.1'")
  
  file_printline(f_qsub, "") #Blank line, make it nicer to read
  
  file_printline(f_qsub, "cd ~/opt/ncr")
  file_printline(f_qsub, "python3 json_train.py {}".format(job_id))
  f_qsub.close()
  
  os.chmod("{}/{}.sh".format(qsub_path, job_id), int('744', 8)) #Make it executable


def generate_qsub_job(qsub_path, job_id, training_args):
  generate_json(qsub_path, job_id, training_args)
  generate_script(qsub_path, job_id)
