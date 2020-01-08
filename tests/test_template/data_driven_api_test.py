#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import requests
from nested_lookup import nested_update

URL_FILE = "url"
RESULT_FILE = "result"

f_url = open(URL_FILE, 'r')
URL_DATA = f_url.readline().rstrip()
f_url.close()

f_result = open(RESULT_FILE, 'r')
RESULT_DATA = json.loads(f_result.read())
f_result.close()

req = requests.get(URL_DATA)
compare_actual = json.loads(req.text)
compare_expected = RESULT_DATA

if "TEST_IGNORE_SCORE" in os.environ:
	if os.environ["TEST_IGNORE_SCORE"] != "":
		compare_actual = nested_update(compare_actual, key='score', value=1.0)
		compare_expected = nested_update(compare_expected, key='score', value=1.0)

try:
	assert compare_actual == compare_expected
except AssertionError as err:
	print("=== BEGIN EXPECTED ===")
	print(compare_expected)
	print("=== END EXPECTED ===")
	
	print("=== BEGIN ACTUAL ===")
	print(compare_actual)
	print("=== END ACTUAL ===")
	if "AUTOTEST" in os.environ:
		if os.environ["AUTOTEST"] != "":
			raise err
	sys.exit(-1)
