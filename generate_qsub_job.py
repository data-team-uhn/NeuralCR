#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import string

import requests
try:
  from requests import HTTPBasicAuth
except ImportError:
  from requests.auth import HTTPBasicAuth

QSUB_WEBDAV_URL = os.environ['QSUB_WEBDAV_URL']
OUTPUT_WEBDAV_URL = os.environ['OUTPUT_WEBDAV_URL']
READY_WEBDAV_URL = os.environ['READY_WEBDAV_URL']
OBO_WEBDAV_URL = os.environ['OBO_WEBDAV_URL']
LOGGING_WEBDAV_URL = os.environ['LOGGING_WEBDAV_URL']
COMPLETE_WEBDAV_URL = os.environ['COMPLETE_WEBDAV_URL']
FAILED_WEBDAV_URL = os.environ['FAILED_WEBDAV_URL']
WEBDAV_CERTPATH = os.environ['WEBDAV_CERTPATH']
WEBDAV_APIKEY = os.environ['WEBDAV_APIKEY']

def file_printline(f, line):
  f.write(line)
  f.write('\n')

def generate_json(qsub_path, job_id, training_args):
  f_json = open("{}/{}.json".format(qsub_path, job_id), 'w')
  file_printline(f_json, training_args.to_json())
  f_json.close()


def filter_safe_chars(s):
  s = str(s)
  out_s = ""
  for c in s:
    if c in (string.ascii_letters + string.digits):
      out_s += c
  return out_s


def generate_script(qsub_path, job_id, given_name):
  f_qsub = open("{}/{}.sh".format(qsub_path, job_id), 'w')
  file_printline(f_qsub, "#!/bin/bash")
  
  file_printline(f_qsub, "") #Blank line, make it nicer to read
  
  file_printline(f_qsub, "#PBS -l mem=32gb,vmem=32gb")
  file_printline(f_qsub, "#PBS -l nodes=1:ppn=2")
  file_printline(f_qsub, "#PBS -l walltime=72:00:00")
  file_printline(f_qsub, "#PBS -j oe")
  file_printline(f_qsub, "#PBS -o $HOME/ncr_logs/{}/".format(job_id))
  
  file_printline(f_qsub, "") #Blank line, make it nicer to read
  
  #Setup the environment variables
  file_printline(f_qsub, "export {}={}".format('QSUB_WEBDAV_URL', QSUB_WEBDAV_URL))
  file_printline(f_qsub, "export {}={}".format('OUTPUT_WEBDAV_URL', OUTPUT_WEBDAV_URL))
  file_printline(f_qsub, "export {}={}".format('READY_WEBDAV_URL', READY_WEBDAV_URL))
  file_printline(f_qsub, "export {}={}".format('OBO_WEBDAV_URL', OBO_WEBDAV_URL))
  file_printline(f_qsub, "export {}={}".format('LOGGING_WEBDAV_URL', LOGGING_WEBDAV_URL))
  
  file_printline(f_qsub, "export {}={}".format('WEBDAV_CERTPATH', WEBDAV_CERTPATH))
  file_printline(f_qsub, "export {}={}".format('WEBDAV_APIKEY', WEBDAV_APIKEY))
  
  file_printline(f_qsub, "") #Blank line, make it nicer to read
  
  file_printline(f_qsub, "module load tensorflow/1.13.1-py3410-cpu")
  file_printline(f_qsub, "pip3 install 'pybind11' --user")
  file_printline(f_qsub, "pip3 install 'fasttext==0.9.1'")
  
  file_printline(f_qsub, "") #Blank line, make it nicer to read
  
  file_printline(f_qsub, "cd ~/opt/ncr")
  file_printline(f_qsub, "python3 json_train.py {}".format(job_id))
  
  file_printline(f_qsub, "") #Blank line, make it nicer to read
  
  #Model training should be complete, upload to WebDAV
  file_printline(f_qsub, "curl --cacert {} --user user:{} {}/{}_config.json -T ~/trained_model_param/{}/config.json --fail && \\".format(WEBDAV_CERTPATH, WEBDAV_APIKEY, OUTPUT_WEBDAV_URL, job_id, job_id))
  file_printline(f_qsub, "curl --cacert {} --user user:{} {}/{}_ncr_weights.h5 -T ~/trained_model_param/{}/ncr_weights.h5 --fail && \\".format(WEBDAV_CERTPATH, WEBDAV_APIKEY, OUTPUT_WEBDAV_URL, job_id, job_id))
  file_printline(f_qsub, "curl --cacert {} --user user:{} {}/{}_onto.json -T ~/trained_model_param/{}/onto.json --fail && \\".format(WEBDAV_CERTPATH, WEBDAV_APIKEY, OUTPUT_WEBDAV_URL, job_id, job_id))
  
  #Notify if this job was a success or a failure
  file_printline(f_qsub, "curl --cacert {} --user user:{} -XPUT -d {} {}/JOBCOMPLETE_{} --fail && exit 0".format(WEBDAV_CERTPATH, WEBDAV_APIKEY, filter_safe_chars(given_name), COMPLETE_WEBDAV_URL, job_id))
  
  file_printline(f_qsub, "") #Blank line, make it nicer to read
  
  #If we have not exited the qsub'd script at this point, something has gone wrong
  file_printline(f_qsub, "curl --cacert {} --user user:{} -XPUT -d {} {}/JOBFAIL_{} --fail".format(WEBDAV_CERTPATH, WEBDAV_APIKEY, "FAILED", FAILED_WEBDAV_URL, job_id))
  
  f_qsub.close()
  
  os.chmod("{}/{}.sh".format(qsub_path, job_id), int('744', 8)) #Make it executable


def generate_qsub_job(qsub_path, job_id, training_args, given_name):
  generate_json(qsub_path, job_id, training_args)
  generate_script(qsub_path, job_id, given_name)

def generate_json_job(job_id, training_args, given_name):
  return json.dumps({"job_id": job_id, "training_args": training_args.to_dict(), "given_name": given_name})
  

def upload_json_job(job_id, training_args, given_name):
  #Upload the training description
  requests.put(QSUB_WEBDAV_URL + "/{}.json".format(job_id), verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY), data=generate_json_job(job_id, training_args, given_name))
  
  #Mark this job as ready to execute
  requests.put(READY_WEBDAV_URL + "/{}".format(job_id), verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY), data='READY')

def download_json_job(qsub_path, obo_path, job_id):
  #Download it...
  resp = requests.get(QSUB_WEBDAV_URL + "/{}.json".format(job_id), verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY))
  resp_obj = json.loads(resp.text)
  
  #...save it...
  f_json = open("{}/{}.json".format(qsub_path, job_id), 'w')
  f_json.write(json.dumps(resp_obj['training_args']))
  f_json.close()
  generate_script(qsub_path, job_id, resp_obj['given_name'])
  
  #...and download the OBO file from WebDAV
  obo_req = requests.get(OBO_WEBDAV_URL + "/{}.obo".format(job_id), verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY))
  f_obo = open("{}/{}.obo".format(obo_path, job_id), 'w')
  f_obo.write(obo_req.text)
  f_obo.close()

if __name__ == '__main__':
  import sys
  import re
  qsub_dir = sys.argv[1]
  obo_dir = sys.argv[2]
  
  #See if there are any jobs that need to be run
  ready_req = requests.get(READY_WEBDAV_URL + "/", verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY))
  ready_jobs = []
  ready_lines = ready_req.text.split('\n')
  for ln in ready_lines:
    if ln.startswith('<li>') and ln.endswith('</li>'):
      matches = re.compile('.+href="(\d+)"').match(ln)
      if len(matches.groups()) == 0:
        continue
      j_id = int(matches.group(1))
      ready_jobs.append(j_id)
      #Delete this job from the READY queue so that it's only exec'd once
      requests.delete(READY_WEBDAV_URL + "/{}".format(j_id), verify=WEBDAV_CERTPATH, auth=HTTPBasicAuth('user', WEBDAV_APIKEY))
  
  print("Will execute the jobs: {}".format(ready_jobs))
  
  for r_jb in ready_jobs:
    download_json_job(qsub_dir, obo_dir, r_jb)
    #Automatically QSUB this job
    os.system("cd {}; qsub {}.sh".format(qsub_dir, r_jb))
  
