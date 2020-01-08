#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import requests

URL_FILE = "url"
RESULT_FILE = "result"

f_url = open(URL_FILE, 'r')
URL_DATA = f_url.readline().rstrip()
f_url.close()

req = requests.get(URL_DATA)
req_data = json.loads(req.text)

f_result = open(RESULT_FILE, 'w')
f_result.write(json.dumps(req_data))
f_result.close()
