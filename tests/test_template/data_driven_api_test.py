#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import requests

URL_FILE = "url"
RESULT_FILE = "result"

f_url = open(URL_FILE, 'r')
URL_DATA = f_url.readline().rstrip()
f_url.close()

f_result = open(RESULT_FILE, 'r')
RESULT_DATA = json.loads(f_result.read())
f_result.close()

req = requests.get(URL_DATA)
try:
	assert json.loads(req.text) == RESULT_DATA
except AssertionError as err:
	print("=== BEGIN EXPECTED ===")
	print(RESULT_DATA)
	print("=== END EXPECTED ===")
	
	print("=== BEGIN ACTUAL ===")
	print(json.loads(req.text))
	print("=== END ACTUAL ===")
	if "AUTOTEST" in os.environ:
		if os.environ["AUTOTEST"] != "":
			raise err
	sys.exit(-1)
