#!flask/bin/python
import json
from functools import wraps
from flask import Flask, jsonify, redirect, request, render_template, url_for, abort
import os
from collections import OrderedDict
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from sent_level import TextAnnotator
import phrase_model
import phraseConfig
import fasttext_reader as reader


app = Flask(__name__)

#'''
rd = reader.Reader(open("data/hp.obo"), True)
model = phrase_model.NCRModel(phraseConfig.Config(), rd)
model.load_params('checkpoints')
textAnt = TextAnnotator(model)
#'''


@app.route('/', methods=['POST'])
def main_page():
    text = request.form['text']
    matches = textAnt.process_text(text, 0.6, True)
    tokens = []
    last = 0
    for match in matches:
        tokens.append({'text':text[last:match[0]]})
        tokens.append({'text':text[match[0]:match[1]], 'hp_id':match[2], 'name':rd.names[match[2]][0]})
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
    curl -i -H "Content-Type: application/json" -X POST -d '{"text":"Retina cancer"}' http://ncr.ccm.sickkids.ca/curr/match/
"""
@app.route('/match/', methods=['POST'])
def match_post(text)
    if not request.json or not 'text' in request.json:
        abort(400)
    res = match(request.json['text'])
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
"""
@app.route('/match/', methods=['GET'])
def match_get(text)
    if not 'text' in request.args:
        abort(400)
    res = match(request.args['text'])
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
    curl -i -H "Content-Type: application/json" -X POST -d '{"text":"The paitient was diagnosed with both cardiac disease and renal cancer."}' http://ncr.ccm.sickkids.ca/curr/annotate/

"""
@app.route('/annotate/', methods=['POST'])
def annotate_post():
    if not request.json or not 'text' in request.json:
        abort(400)
    res = annotate(request.json['text'])
    return jsonify(res)
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
Date: Sun, 23 Apr 2017 17:51:51 GMT
Server: Apache/2.4.18 (Ubuntu)
Content-Length: 611
Content-Type: application/json

[
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
"""
@app.route('/annotate/', methods=['GET'])
def annotate_get():
    if not 'text' in request.args:
        abort(400)
    res = annotate(request.args['text'])
    return jsonify(res)

def match(text)
    matches = model.get_hp_id([text], 10)[0]
    res = []
    for x in matches:
        tmp = OrderedDict([('hp_id',x[0]),
                ('names',rd.names[x[0]]), 
                ('score',str(x[1]))])
        res.append(tmp)
    return {"matches":res}

def annotate(text):
    matches = textAnt.process_text(text, 0.6, True)
    res = []
    for x in matches:
        tmp = OrderedDict([('start',x[0]),
                ('end',x[1]),
                ('hp_id',x[2]),
                ('names',rd.names[x[2]]), 
                ('score',str(x[3]))])
        res.append(tmp)
    return {"matches":res}


if __name__ == '__main__':
    print "Model loaded"
    app.run()

