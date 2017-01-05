#!flask/bin/python
from flask import Flask, jsonify
import phrase_annotator

app = Flask(__name__)
ant = phrase_annotator.create_annotator("/home/aryan/codes/NeuralCR/checkpoints", "data/", True, False)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/phrase/<string:querry>', methods=['GET'])
def get_task(querry):
    pquerry = querry.replace('+',' ')
    matches = ant.get_hp_id([pquerry],5)[0]
    res = []
    for x in matches:
#        sys.stdout.write(x[0]+' '+str(tools['ant'].rd.names[x[0]])+' '+str(x[1])+'\n')
        tmp = {'hp_id':x[0],
                'names':ant.rd.names[x[0]], 
                'score':str(1.0-x[1])}
        res.append(tmp)

    return jsonify(res)
    #return matches[0][0][0]
    #return jsonify(tmp)

if __name__ == '__main__':
    print "Done loading model!"
    app.run()
