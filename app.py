#!flask/bin/python
import json

from functools import wraps
from flask import Flask, jsonify, redirect, request, render_template, url_for, abort, Response
import os
import re
from collections import OrderedDict
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import requests
try:
  from requests import HTTPBasicAuth
except ImportError:
  from requests.auth import HTTPBasicAuth

import pycurl

import ncrmodel
import train
from generate_qsub_job import upload_json_job

CONST_HOMEDIR = os.environ['HOME']

CONST_FASTTEXT_WORD_VECTOR_FILEPATH = "{}/opt/ncr/model_params/pmc_model_new.bin" #Relative to $HOME
CONST_NEGFILE_FILEPATH = "{}/wikipedia_small.txt" #Relative to $HOME
CONST_UPLOADED_OBO_DIR = "{}/uploaded_obo" #Relative to $HOME
CONST_PARAMS_FILEPATH = "{}/trained_model_param" #Relative to $HOME
CONST_QSUB_FILEPATH = "{}/qsub" #Relative to $HOME

OBO_WEBDAV_URL = os.environ['OBO_WEBDAV_URL']
LOGGING_WEBDAV_URL = os.environ['LOGGING_WEBDAV_URL']
COMPLETE_WEBDAV_URL = os.environ['COMPLETE_WEBDAV_URL']
FAILED_WEBDAV_URL = os.environ['FAILED_WEBDAV_URL']
OUTPUT_WEBDAV_URL = os.environ['OUTPUT_WEBDAV_URL']
WEBDAV_CERTPATH = os.environ['WEBDAV_CERTPATH']
WEBDAV_APIKEY = os.environ['WEBDAV_APIKEY']

app = Flask(__name__)

#Stored in a form of {"object": model object, "threshold": threshold value}
NCR_MODELS = {}

"""
Start with the two freely available pre-trained NCR models. This
NCR_MODELS data structure will be populated with additional models as
training jobs are submitted. Note that, unlike previous versions of this
application, only the model constructor arguments are stored as storing
more than a few trained models would exceed the memory limitations of
modern computers. Trained model objects are constructed on an as-needed
basis.
"""
NCR_MODELS['HPO'] = {}
NCR_MODELS['HPO']['object'] = ('model_params/0', 'model_params/pmc_model_new.bin')
NCR_MODELS['HPO']['threshold'] = 0.6

NCR_MODELS['MONDO'] = {}
NCR_MODELS['MONDO']['object'] = ('model_params/1', 'model_params/pmc_model_new.bin')
NCR_MODELS['MONDO']['threshold'] = 0.6 #Just a copy+paste, should have better reasoning for selecting this value

AVAILABLE_MODEL_ID = []
def update_ncr_model_list():
  if 'AUTOTEST' in os.environ:
    if len(os.environ['AUTOTEST']) != 0:
      return
   
  #Check for complete jobs under /complete in WebDAV
  complete_req = requests.get(COMPLETE_WEBDAV_URL + "/", verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY))
  complete_lines = complete_req.text.split('\n')
  completed_training_jobs = []
  for cl in complete_lines:
    if cl.startswith('<li>') and cl.endswith('</li>') and "JOBCOMPLETE_" in cl:
      matches = re.compile('.+JOBCOMPLETE\_(\d+)').match(cl)
      if len(matches.groups()) == 0:
        continue
      this_model_id = int(matches.group(1))
      if this_model_id not in AVAILABLE_MODEL_ID:
        completed_training_jobs.append(this_model_id)
  
  print("completed_training_jobs = {}".format(completed_training_jobs))
  
  #For each completed job...
  for cj in completed_training_jobs:
    #...download the trained model
    os.mkdir("new_model_params/{}".format(cj))
    for fname in ["config.json", "ncr_weights.h5", "onto.json"]:
      with open("new_model_params/{}/{}".format(cj, fname), 'wb') as f:
        print("Getting new_model_params/{}/{}...".format(cj, fname))
        c = pycurl.Curl()
        c.setopt(c.URL, OUTPUT_WEBDAV_URL + "/{}_{}".format(cj, fname))
        c.setopt(c.WRITEDATA, f)
        c.setopt(c.CAINFO, WEBDAV_CERTPATH)
        c.setopt(c.USERPWD, "user:{}".format(WEBDAV_APIKEY))
        c.perform()
        c.close()
    
    #...construct the NCR() object
    #...get the given name for this model
    name_req = requests.get(COMPLETE_WEBDAV_URL + "/JOBCOMPLETE_{}".format(cj), verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY))
    new_model_name = name_req.text.rstrip()
    NCR_MODELS[new_model_name] = {}
    NCR_MODELS[new_model_name]['object'] = ("new_model_params/{}".format(cj), 'model_params/pmc_model_new.bin')
    NCR_MODELS[new_model_name]['threshold'] = 0.6 #Just a copy+paste, should have better reasoning for selecting this value
    
    #Don't re-download
    AVAILABLE_MODEL_ID.append(cj)

#On startup, load all models from WebDAV
update_ncr_model_list()

running_job_id = 0
def generate_job_id():
  global running_job_id
  assign_job_id = running_job_id
  running_job_id += 1
  return assign_job_id

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
    update_ncr_model_list()
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
    if selected_model not in NCR_MODELS:
        abort(400)
    del NCR_MODELS[selected_model]
    return jsonify({'status': 'success'})


#Serve the webpage for model training
@app.route('/submit_training_job/', methods=['GET'])
def submit_training_job_get():
    return render_template("submit_training_job.html")

#Receive the upload form including the OBO ontology file for model training
@app.route('/submit_training_job/', methods=['POST'])
def submit_training_job_post():
    if 'ontology' not in request.files:
      abort(400)
    
    if 'name' not in request.form:
      abort(400)
    
    #Generate a JOB ID for this training task
    j_id = generate_job_id()
    
    ontology_file = request.files['ontology']
    ontology_filepath = "{}/{}.obo".format(CONST_UPLOADED_OBO_DIR, j_id)
    ontology_file.save(ontology_filepath.format(CONST_HOMEDIR))
    
    #Upload this ontology file to WebDAV
    fdata = open(ontology_filepath.format(CONST_HOMEDIR), 'rb')
    requests.put(OBO_WEBDAV_URL + "/{}.obo".format(j_id), verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY), data=fdata)
    fdata.close()
    
    #Start the training
    print("[JOB: {}] Queue'd training model {}, at root={}...".format(j_id, request.form['name'], request.form['oboroot']))
    params_output_dir = CONST_PARAMS_FILEPATH + "/{}/".format(j_id)
    training_proc_args = train.MainTrainArgClass(
      obofile=ontology_filepath,
      oboroot=request.form['oboroot'],
      fasttext=CONST_FASTTEXT_WORD_VECTOR_FILEPATH,
      neg_file=CONST_NEGFILE_FILEPATH,
      output=params_output_dir,
      verbose=True
      )
    
    upload_json_job(j_id, training_proc_args, request.form['name'])
    return jsonify({'status': 'submitted', 'id': j_id})


@app.route('/log/<int:j_id>')
def get_job_logs(j_id):
    #query the WebDAV server
    jobs_query = requests.get(LOGGING_WEBDAV_URL, verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY))
    job_lines = jobs_query.text.split('\n')
    selected_ids = []
    for jl in job_lines:
        if jl.startswith('<li>') and jl.endswith('</li>'):
            matches = re.compile('.+(\d+)\_(\d+)').match(jl)
            if len(matches.groups()) == 0:
                continue
            line_jobid = int(matches.group(1))
            line_messageid = int(matches.group(2))
            if line_jobid == j_id:
                selected_ids.append(line_messageid)
    selected_ids.sort()
    
    #Download all messages
    saved_messages = ""
    for message_id in selected_ids:
      get_url = LOGGING_WEBDAV_URL + "/{}_{}.logmsg".format(j_id, message_id)
      req = requests.get(get_url, verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY))
      saved_messages += "{}\n".format(req.text)
    return Response(saved_messages, mimetype='text/plain')

@app.route('/job/<int:j_id>')
def get_job_status(j_id):
    #if j_id is >= running_job_id then j_id is invalid
    if j_id >= running_job_id:
        return jsonify({'status': 'invalid'})
    
    #query the WebDAV server - is it in the COMPLETE directory
    complete_query = requests.get(COMPLETE_WEBDAV_URL + "/", verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY))
    complete_query_lines = complete_query.text.split('\n')
    for ln in complete_query_lines:
        if '"JOBCOMPLETE_{}"'.format(j_id) in ln:
            return jsonify({'status': 'complete'})
    
    #query the WebDAV server - is it in the FAILED directory
    failed_query = requests.get(FAILED_WEBDAV_URL + "/", verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY))
    failed_query_lines = failed_query.text.split('\n')
    for ln in failed_query_lines:
        if '"JOBFAIL_{}"'.format(j_id) in ln:
            return jsonify({'status': 'failed'})
    
    #otherwise, assume the job is submitted (and running/queue'd)
    return jsonify({'status': 'submitted'})

"""
@api {post} /match/ POST Method
@apiName PostMatch
@apiGroup match
@apiVersion 0.1.0

@apiDescription
Returns a list of concept classes from the ontology that best match the input term.

@apiParam {String} text Input term

@apiSuccess {Object[]} matches list of top matching classes
@apiSuccess {String} matches.hp_id  Class identifier
@apiSuccess {String[]} matches.names  List of names and synonyms of the matched class
@apiSuccess {Double} matches.score Matching score (a probability between 0 and 1)

@apiExample {curl} Example usage:
    curl -i -H "Content-Type: application/json" -X POST -d '{"text":"Retina cancer", "model":"HPO"}' http://ncr.ccm.sickkids.ca/curr/match/
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
@apiSuccess {String} matches.hp_id  Class identifier
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
@apiSuccess {String} matches.hp_id  Class identifier
@apiSuccess {String[]} matches.names  List of names and synonyms of the matched class
@apiSuccess {Double} matches.score Matching score (a probability between 0 and 1)

@apiExample {curl} Example usage:
    curl -i -H "Content-Type: application/json" -X POST -d '{"text":"The patient was diagnosed with both cardiac disease and renal cancer.", "model": "HPO"}' http://ncr.ccm.sickkids.ca/curr/annotate/

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
            return jsonify(res)
        #Multiple models are available, but none specified
        abort(400)
    if request.json['model'] not in NCR_MODELS:
        abort(400)
    res = annotate(NCR_MODELS[request.json['model']]['object'], NCR_MODELS[request.json['model']]['threshold'], request.json['text'])
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
@apiSuccess {String} matches.hp_id  Class identifier
@apiSuccess {String[]} matches.names  List of names and synonyms of the matched class
@apiSuccess {Double} matches.score Matching score (a probability between 0 and 1)

@apiExample {curl} Example usage:
    curl -i http://ncr.ccm.sickkids.ca/curr/annotate/?text=The+patient+was+diagnosed+with+both+cardiac+disease+and+renal+cancer.

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
            return jsonify(res)
        #Multiple models are available, but none specified
        abort(400)
    if request.args['model'] not in NCR_MODELS:
        abort(400)
    if not 'text' in request.args:
        abort(400)
    res = annotate(NCR_MODELS[request.args['model']]['object'], NCR_MODELS[request.args['model']]['threshold'], request.args['text'])
    return jsonify(res)

def match(model, text):
    model = ncrmodel.NCR.loadfromfile(*model)
    matches = model.get_match([text], 10)[0]
    res = []
    for x in matches:
        tmp = OrderedDict([('hp_id',x[0]),
                ('names',model.ont.names[x[0]]),
                ('score',str(x[1]))])

        res.append(tmp)
    return {"matches":res}

def annotate(model, threshold, text):
    model = ncrmodel.NCR.loadfromfile(*model)
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


if __name__ == '__main__':
    print("Model loaded")
    app.run(host='0.0.0.0')

