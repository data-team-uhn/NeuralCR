#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import string
from itertools import groupby

"""
Generates a list of index tuples for the words within a text.
"""
def splitWithIndicesNoPunct(s, c=' '):
    p = 0
    for k, g in groupby(s, lambda x:x==c):
        q = p + sum(1 for i in g)
        if not k:
            yield p, p + len(s[p:q].rstrip(string.punctuation))
        p = q


class OntologyInterface:
    def __init__(self, names):
        self.names = names

class ClassicModel:
    def __init__(self, id_file, title_file):
        with open(id_file, 'r') as f:
            self.id_map = json.loads(f.read())
        with open(title_file, 'r') as f:
            self.ont = OntologyInterface(json.loads(f.read()))

    def get_match(self, query, count=1):
        res = [[]]
        queryWord = query[0].upper()
        if queryWord in self.id_map:
            res[0].append([self.id_map[queryWord], 1.0])

        return res

    def annotate_text(self, text, threshold=0.8):
        res = []
        for wordStart, wordEnd in splitWithIndicesNoPunct(text):
            word = text[wordStart:wordEnd]

            #Convert to upper-case
            word = word.upper()

            #Check if this is a known term
            if word not in self.id_map:
                continue

            res.append([wordStart, wordEnd, self.id_map[word], 1.0])

        return res
