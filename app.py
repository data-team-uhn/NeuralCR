#!flask/bin/python
from flask import Flask, jsonify
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from sent_level import TextAnnotator
import phrase_model
import phraseConfig
import fasttext_reader as reader


app = Flask(__name__)

rd = reader.Reader("data", True)
model = phrase_model.NCRModel(phraseConfig.Config(), rd)
model.load_params('checkpoints')
textAnt = TextAnnotator(model)

@app.route('/term/<string:querry>', methods=['GET'])
def get_task_phrase(querry):
    pquerry = querry.replace('+',' ')
    matches = model.get_hp_id([pquerry], 10)[0]
    res = []
    for x in matches:
        tmp = {'hp_id':x[0],
                'names':rd.names[x[0]], 
                'score':str(x[1])}
        res.append(tmp)
    return jsonify(res)

@app.route('/text/<string:querry>', methods=['GET'])
def get_task_text(querry):
    pquerry = querry.replace('+',' ')
    matches = textAnt.process_text(pquerry, 0.6, True)
    res = []
    for x in matches:
        tmp = { 'start':x[0],
                'end':x[1],
                'hp_id':x[2],
                'names':rd.names[x[2]], 
                'score':str(x[3])}
        res.append(tmp)
    return jsonify(res)


if __name__ == '__main__':
    print "Model loaded"
    app.run()

