#!flask/bin/python
import json

from functools import wraps
from flask import Flask, jsonify, redirect, request, render_template, url_for, abort
from werkzeug.utils import secure_filename
import os
import argparse
from collections import OrderedDict
from urllib.parse import parse_qs
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import ncrmodel
import basic_text_matcher

cli_arg_parser = argparse.ArgumentParser()
cli_arg_parser.add_argument("--allow_model_delete",
    help="Allows NCR models to be deleted via HTTP DELETE method",
    action="store_true"
    )
cli_arg_parser.add_argument("--allow_model_put",
    help="Allows NCR models to be added based on files present on this server",
    action="store_true"
    )
cli_arg_parser.add_argument("--always_prefix_model_path",
    help="Always prefix the model path to the annotated results, even \
    if multiple trained models are not specified",
    action="store_true")

CLI_ARGS = cli_arg_parser.parse_args()

TRAINED_MODELS_PATH = "model_params/"

app = Flask(__name__)

#Stored in a form of {"object": model object, "threshold": threshold value}
NCR_MODELS = {}

"""
Start with at least one hard-coded model, in future versions of this API, newer trained models
can be automatically added to the NCR_MODELS data structure. For now, simply modify the following
lines to select which trained models are to be available for use.
"""
NCR_MODELS['HPO'] = {}
NCR_MODELS['HPO']['object'] = ncrmodel.NCR.loadfromfile('checks', '../NeuralCR/data/model_pmc.bin')
NCR_MODELS['HPO']['threshold'] = 0.6


@app.route('/', methods=['POST'])
def main_page():
    text = request.form['text']
    matches = model.annotate_text(text, threshold)
    matches = sorted(matches, key=lambda x: x[0])
    tokens = []
    last = 0
    for match in matches:
        tokens.append({'text':text[last:match[0]]})
        tokens.append({'text':text[match[0]:match[1]], 'hp_id':match[2], 'name':model.ont.names[match[2]][0]})
        last = match[1]
    tokens.append({'text':text[last:]})
    return render_template("main.html", texxt=text, tokens=tokens)

@app.route('/')
def form():
    return render_template("main.html", texxt= "The patient was diagnosed with retina cancer.")

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


@app.route('/models/', methods=['GET'])
def ls_models():
    new_mapping = {}
    for k in NCR_MODELS.keys():
        new_mapping[k] = {}
        for param in NCR_MODELS[k].keys():
            if param == 'object':
                continue #Can't serialize this to JSON
            new_mapping[k][param] = NCR_MODELS[k][param]
    return jsonify(new_mapping)

@app.route('/models/<selected_model>', methods=['DELETE'])
def delete_model(selected_model):
    if not CLI_ARGS.allow_model_delete:
        abort(400)
    if selected_model not in NCR_MODELS:
        abort(400)
    del NCR_MODELS[selected_model]
    return jsonify({'status': 'success'})

@app.route('/models/<selected_model>', methods=['PUT'])
def put_model(selected_model):
    if not CLI_ARGS.allow_model_put:
        abort(400)
    if not request.json:
        abort(400)
    if 'model_type' not in request.json:
        abort(400)
    if type(request.json['model_type']) != str:
        abort(400)
    if request.json['model_type'] not in ['neural', 'basic']:
        abort(400)

    if request.json['model_type'] == 'neural':
        for arg in ['param_dir', 'word_model_file']:
            if arg not in request.json:
                abort(400)
            if type(request.json[arg]) != str:
                abort(400)

        if 'threshold' not in request.json:
            abort(400)

        if type(request.json['threshold']) != float:
            abort(400)

        param_dir = os.path.join(TRAINED_MODELS_PATH, secure_filename(request.json['param_dir']))
        word_model_file = os.path.join(TRAINED_MODELS_PATH, secure_filename(request.json['word_model_file']))
        NCR_MODELS[selected_model] = {}
        NCR_MODELS[selected_model]['object'] = ncrmodel.NCR.safeloadfromjson(param_dir, word_model_file)
        NCR_MODELS[selected_model]['threshold'] = request.json['threshold']

    elif request.json['model_type'] == 'basic':
        for arg in ['id_file', 'title_file']:
            if arg not in request.json:
                abort(400)
            if type(request.json[arg]) != str:
                abort(400)

        id_file = os.path.join(TRAINED_MODELS_PATH, secure_filename(request.json['id_file']))
        title_file = os.path.join(TRAINED_MODELS_PATH, secure_filename(request.json['title_file']))
        NCR_MODELS[selected_model] = {}
        NCR_MODELS[selected_model]['object'] = basic_text_matcher.BasicTextMatcher(id_file, title_file)
        NCR_MODELS[selected_model]['threshold'] = 1.0

    return jsonify({'status': 'success'})

"""
@api {post} /match/ POST Method
@apiName PostMatch
@apiGroup match
@apiVersion 0.1.0

@apiDescription
Returns a list of concept classes from the ontology that best match the input term.

@apiParam {String} text Input term

@apiSuccess {Object[]} matches list of top matching classes
@apiSuccess {String} matches.hp_id  Class identifer
@apiSuccess {String[]} matches.names  List of names and synonyms of the matched class
@apiSuccess {Double} matches.score Matching score (a probability between 0 and 1)

@apiExample {curl} Example usage:
    curl -i -H "Content-Type: application/json" -X POST -d '{"text":"Retina cancer", "model":"HPO"}' http://ncr.ccm.sickkids.ca/curr/match/
    curl -i -H "Content-Type: application/json" -X POST -d '{"text":"Retina cancer"}' http://ncr.ccm.sickkids.ca/curr/match/
"""
@app.route('/match/', methods=['POST'])
def match_post():
    if not request.json:
        abort(400)
    if not 'text' in request.json:
        abort(400)
    if not 'model' in request.json:
        #Maintain backwards compatibility
        if len(list(NCR_MODELS.keys())) == 1:
            fallback_key = list(NCR_MODELS.keys())[0]
            res = match(NCR_MODELS[fallback_key]['object'], request.json['text'])
            return jsonify(res)
        #Multiple models are available, but none specified
        abort(400)
    if request.json['model'] not in NCR_MODELS:
        abort(400)
    res = match(NCR_MODELS[request.json['model']]['object'], request.json['text'])
    return jsonify(res)

"""
@api {get} /match/ GET Method
@apiName GetMatch
@apiGroup match
@apiVersion 0.1.0

@apiDescription
Returns a ranked list of top concept classes from the ontology that best match the input term.

@apiParam {String} text Input term.

@apiSuccess {Object[]} matches list of top matching classes
@apiSuccess {String} matches.hp_id  Class identifer
@apiSuccess {String[]} matches.names  List of names and synonyms of the matched class
@apiSuccess {Double} matches.score Matching score (a probability between 0 and 1)

@apiExample {curl} Example usage:
    curl -i http://ncr.ccm.sickkids.ca/curr/match/?text=retina+cancer

@apiSuccessExample {json} Success-Response:
HTTP/1.1 200 OK
Date: Sun, 23 Apr 2017 18:47:40 GMT
Server: Apache/2.4.18 (Ubuntu)
Content-Length: 1745
Content-Type: application/json

{
  "matches": [
    {
      "hp_id": "HP:0012777", 
      "names": [
        "Retinal neoplasm"
      ], 
      "score": "0.708877"
    }, 
    {
      "hp_id": "HP:0100012", 
      "names": [
        "Neoplasm of the eye", 
        "Neoplasia of the eye"
      ], 
      "score": "0.161902"
    }, 
    {
      "hp_id": "HP:0100006", 
      "names": [
        "Neoplasm of the central nervous system", 
        "Neoplasia of the central nervous system", 
        "Tumors of the central nervous system"
      ], 
      "score": "0.0105245"
    }, 
    {
      "hp_id": "HP:0007716", 
      "names": [
        "Intraocular melanoma", 
        "Uveal melanoma"
      ], 
      "score": "0.00986601"
    }, 
    {
      "hp_id": "HP:0000479", 
      "names": [
        "Abnormality of the retina", 
        "Abnormal retina", 
        "Anomaly of the retina", 
        "Retinal disease"
      ], 
      "score": "0.00977207"
    }, 
    {
      "hp_id": "HP:0004375", 
      "names": [
        "Neoplasm of the nervous system", 
        "Neoplasia of the nervous system", 
        "Tumor of the nervous system"
      ], 
      "score": "0.00862639"
    }, 
    {
      "hp_id": "HP:0009919", 
      "names": [
        "Retinoblastoma"
      ], 
      "score": "0.00727"
    }, 
    {
      "hp_id": "HP:0030692", 
      "names": [
        "Brain neoplasm", 
        "Brain tumor", 
        "Brain tumour"
      ], 
      "score": "0.00584657"
    }, 
    {
      "hp_id": "HP:0100836", 
      "names": [
        "Malignant neoplasm of the central nervous system"
      ], 
      "score": "0.0049508"
    }, 
    {
      "hp_id": "HP:0001098", 
      "names": [
        "Abnormality of the fundus"
      ], 
      "score": "0.00389818"
    }
  ]
}
"""
@app.route('/match/', methods=['GET'])
def match_get():
    if not 'model' in request.args:
        #Maintain backwards compatibility
        if len(list(NCR_MODELS.keys())) == 1:
            fallback_key = list(NCR_MODELS.keys())[0]
            res = match(NCR_MODELS[fallback_key]['object'], request.args['text'])
            return jsonify(res)
        #Multiple models are available, but none specified
        abort(400)
    if request.args['model'] not in NCR_MODELS:
        abort(400)
    if not 'text' in request.args:
        abort(400)
    res = match(NCR_MODELS[request.args['model']]['object'], request.args['text'])
    return jsonify(res)

"""
@api {post} /annotate/ POST Method
@apiName PostAnnotate
@apiGroup annotate
@apiVersion 0.1.0

@apiDescription
Annotates an input text with concepts from the ontology. Returns the clauses that match an ontology class.

@apiParam {String} text Input text.

@apiSuccess {Object[]} matches List of recognized concepts
@apiSuccess {Integer} matches.start  Start location of the match in text
@apiSuccess {Integer} matches.end  End location of the match in text
@apiSuccess {String} matches.hp_id  Class identifer
@apiSuccess {String[]} matches.names  List of names and synonyms of the matched class
@apiSuccess {Double} matches.score Matching score (a probability between 0 and 1)

@apiExample {curl} Example usage:
    curl -i -H "Content-Type: application/json" -X POST -d '{"text":"The paitient was diagnosed with both cardiac disease and renal cancer.", "model": "HPO"}' http://ncr.ccm.sickkids.ca/curr/annotate/
    curl -i -H "Content-Type: application/json" -X POST -d '{"text":"The paitient was diagnosed with both cardiac disease and renal cancer."}' http://ncr.ccm.sickkids.ca/curr/annotate/

"""
@app.route('/annotate/', methods=['POST'])
def annotate_post():
    if not request.json:
        abort(400)
    if not 'text' in request.json:
        abort(400)
    if not 'model' in request.json:
        #Maintain backwards compatibility
        if len(list(NCR_MODELS.keys())) == 1:
            fallback_key = list(NCR_MODELS.keys())[0]
            res = annotate(NCR_MODELS[fallback_key]['object'], NCR_MODELS[fallback_key]['threshold'], request.json['text'])
            return jsonify(prefix_model_path(res, fallback_key))
        #Multiple models are available, but none specified
        abort(400)
    if type(request.json['model']) == str:
        if request.json['model'] not in NCR_MODELS:
            abort(400)
        res = annotate(NCR_MODELS[request.json['model']]['object'], NCR_MODELS[request.json['model']]['threshold'], request.json['text'])
        return jsonify(prefix_model_path(res, request.json['model']))
    elif type(request.json['model']) == list:
        matches = []
        for model in request.json['model']:
            if model not in NCR_MODELS:
                abort(400)
            res = annotate(NCR_MODELS[model]['object'], NCR_MODELS[model]['threshold'], request.json['text'])
            for match in res['matches']:
                if CLI_ARGS.always_prefix_model_path:
                    #Prefix the 'hp_id' associated value with its vocabulary path
                    match['hp_id'] = "/Vocabularies/{}/".format(model) + match['hp_id']
                matches.append(match)
        return jsonify({'matches': matches})
"""
@api {get} /annotate/ GET Method
@apiName GetAnnotate
@apiGroup annotate
@apiVersion 0.1.0
@apiDescription
Annotates an input text with concepts from the ontology. Returns the clauses that match an ontology class.

@apiParam {String} text Input text.

@apiSuccess {Object[]} matches List of recognized concepts
@apiSuccess {Integer} matches.start  Start location of the match in text
@apiSuccess {Integer} matches.end  End location of the match in text
@apiSuccess {String} matches.hp_id  Class identifer
@apiSuccess {String[]} matches.names  List of names and synonyms of the matched class
@apiSuccess {Double} matches.score Matching score (a probability between 0 and 1)

@apiExample {curl} Example usage:
    curl -i http://ncr.ccm.sickkids.ca/curr/annotate/?text=The+paitient+was+diagnosed+with+both+cardiac+disease+and+renal+cancer.

@apiSuccessExample {json} Success-Response:
HTTP/1.1 200 OK
Date: Sun, 23 Apr 2017 18:46:33 GMT
Server: Apache/2.4.18 (Ubuntu)
Content-Length: 686
Content-Type: application/json

{
  "matches": [
    {
      "end": 52, 
      "hp_id": "HP:0001627", 
      "names": [
        "Abnormal heart morphology", 
        "Abnormality of cardiac morphology", 
        "Abnormality of the heart", 
        "Cardiac abnormality", 
        "Cardiac anomalies", 
        "Congenital heart defect", 
        "Congenital heart defects"
      ], 
      "score": "0.696756", 
      "start": 37
    }, 
    {
      "end": 69, 
      "hp_id": "HP:0009726", 
      "names": [
        "Renal neoplasm", 
        "Kidney cancer", 
        "Neoplasia of the kidneys", 
        "Renal neoplasia", 
        "Renal tumors"
      ], 
      "score": "0.832163", 
      "start": 45
    }
  ]
}
"""
@app.route('/annotate/', methods=['GET'])
def annotate_get():
    if not 'model' in request.args:
        #Maintain backwards compatibility
        if len(list(NCR_MODELS.keys())) == 1:
            fallback_key = list(NCR_MODELS.keys())[0]
            res = annotate(NCR_MODELS[fallback_key]['object'], NCR_MODELS[fallback_key]['threshold'], request.args['text'])
            return jsonify(prefix_model_path(res, fallback_key))
        #Multiple models are available, but none specified
        abort(400)
    if request.args['model'] not in NCR_MODELS:
        abort(400)
    if not 'text' in request.args:
        abort(400)
    model_list = parse_qs(request.query_string.decode()).get('model', None)
    matches = []
    for model in model_list:
        if model not in NCR_MODELS:
            abort(400)
        res = annotate(NCR_MODELS[model]['object'], NCR_MODELS[model]['threshold'], request.args['text'])
        for match in res['matches']:
            if CLI_ARGS.always_prefix_model_path:
                #Prefix the 'hp_id' associated value with its vocabulary path
                match['hp_id'] = "/Vocabularies/{}/".format(model) + match['hp_id']
            matches.append(match)
    return jsonify({'matches': matches})

def match(model, text):
    matches = model.get_match([text], 10)[0]
    res = []
    for x in matches:
        tmp = OrderedDict([('hp_id',x[0]),
                ('names',model.ont.names[x[0]]), 
                ('score',str(x[1]))])

        res.append(tmp)
    return {"matches":res}

def annotate(model, threshold, text):
    matches = model.annotate_text(text, threshold)
    res = []
    for x in matches:
        tmp = OrderedDict([('start',x[0]),
                ('end',x[1]),
                ('hp_id',x[2]),
                ('names',model.ont.names[x[2]]), 
                ('score',str(x[3]))])
        res.append(tmp)
    return {"matches":res}

def prefix_model_path(ncroutput, model_name):
    if not CLI_ARGS.always_prefix_model_path:
        return ncroutput
    new_matches = []
    for match in ncroutput['matches']:
        new_match = match
        new_match['hp_id'] = "/Vocabularies/{}/".format(model_name) + new_match['hp_id']
        new_matches.append(new_match)
    res = ncroutput
    res['matches'] = new_matches
    return res

if __name__ == '__main__':
    print("Model loaded")
    app.run()

