#!/usr/bin/python

import os
import ncrmodel
from werkzeug.utils import secure_filename

def loadfromrequest(request, trained_models_path):
    for arg in ['param_dir', 'word_model_file']:
        if arg not in request.json:
            raise Exception("Missing parameters")
        if type(request.json[arg]) != str:
            raise Exception("Improper parameter data type")

    if 'threshold' not in request.json:
        raise Exception("Missing parameters")

    if type(request.json['threshold']) != float:
        raise Exception("Improper parameter data type")

    param_dir = os.path.join(trained_models_path, secure_filename(request.json['param_dir']))
    word_model_file = os.path.join(trained_models_path, secure_filename(request.json['word_model_file']))
    this_model = {}
    this_model['object'] = ncrmodel.NCR.safeloadfromjson(param_dir, word_model_file)
    this_model['threshold'] = request.json['threshold']
    return this_model
