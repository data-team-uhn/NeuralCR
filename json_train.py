#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trains an NCR model based on training data associated with the
given JOB ID.
"""

import os
import sys
import json

import train

CONST_HOMEDIR = os.environ['HOME']

CONST_QSUB_FILEPATH = "{}/qsub".format(os.environ['HOME'])
JOB_ID = int(sys.argv[1])

#Set CWD
os.chdir(CONST_QSUB_FILEPATH)

#Load the JSON file associated with this
f_json = open("{}.json".format(JOB_ID), 'r')
training_args_dict = json.loads(f_json.read())
f_json.close()

#Resolve the $HOME directory
for key in training_args_dict.keys():
  if type(training_args_dict[key]) == str:
    training_args_dict[key] = training_args_dict[key].format(CONST_HOMEDIR)

#Log info
print("Running NCR training with params:")
for key in training_args_dict.keys():
  print("\t{} --> {}".format(key, training_args_dict[key]))

#Start the training
training_args = train.MainTrainArgClass(**training_args_dict)
train.main_train(training_args)
