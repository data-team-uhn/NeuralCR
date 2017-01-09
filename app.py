#!flask/bin/python
from flask import Flask, jsonify
from sent_level import TextAnnotator
import phrase_annotator

app = Flask(__name__)
ant = phrase_annotator.create_annotator("/home/aryan/codes/NeuralCR/checkpoints", "data/", True, False)
textAnt = TextAnnotator(ant=ant)

@app.route('/term_match/<string:querry>', methods=['GET'])
def get_task_phrase(querry):
    pquerry = querry.replace('+',' ')
    matches = ant.get_hp_id([pquerry], 10)[0]
    res = []
    for x in matches:
        tmp = {'hp_id':x[0],
                'names':ant.rd.names[x[0]], 
                'score':1.0 - x[1]}
        res.append(tmp)
    return jsonify(res)

@app.route('/text_annotate/<string:querry>', methods=['GET'])
def get_task_text(querry):
    pquerry = querry.replace('+',' ')
    matches = textAnt.process_text(pquerry, 0.5, True)
    res = []
    for x in matches:
        tmp = { 'start':x[0],
                'end':x[1],
                'hp_id':x[2],
                'names':ant.rd.names[x[2]], 
                'score':1.0 - x[3]}
        res.append(tmp)
    return jsonify(res)


if __name__ == '__main__':
    print "Model loaded"
    app.run()

