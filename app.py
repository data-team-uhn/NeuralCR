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

'''
rd = reader.Reader(open("data/hp.obo"), True)
model = phrase_model.NCRModel(phraseConfig.Config(), rd)
model.load_params('checkpoints')
textAnt = TextAnnotator(model)
'''


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


@app.route('/match/', methods=['GET', 'POST'])
def get_task_phrase():
    if request.method == 'POST':
        if not request.json or not 'text' in request.json:
            abort(400)
	pquerry = request.json['text']
    else:
        if not 'text' in request.args:
            abort(400)
	pquerry = request.args['text']
    matches = model.get_hp_id([pquerry], 10)[0]
    res = []
    for x in matches:
        tmp = OrderedDict([('hp_id',x[0]),
                ('names',rd.names[x[0]]), 
                ('score',str(x[1]))])
        res.append(tmp)
    return jsonify(res)

@app.route('/annotate/', methods=['GET', 'POST'])
def get_task_text():
    if request.method == 'POST':
        if not request.json or not 'text' in request.json:
            abort(400)
	pquerry = request.json['text']
    else:
        if not 'text' in request.args:
            abort(400)
	pquerry = request.args['text']

    matches = textAnt.process_text(pquerry, 0.6, True)
    res = []
    for x in matches:
        tmp = OrderedDict([('start',x[0]),
                ('end',x[1]),
                ('hp_id',x[2]),
                ('names',rd.names[x[2]]), 
                ('score',str(x[3]))])
        res.append(tmp)
    return jsonify(res)


if __name__ == '__main__':
    print "Model loaded"
    app.run()

