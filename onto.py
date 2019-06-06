import numpy as np
import re

def _get_tag_value(line):
  line = line[:line.find('!')]
  col_index = line.find(':')
  if col_index == -1:
    return "", ""
  tag = line[:col_index]
  quotes=re.findall(r'\"(.+?)\"',line)
  if len(quotes) == 0:
    value = line[col_index+1:].strip()
  else:
    value = quotes[0].strip()
  return tag, value

def _dfs(c, kids, mark):
  mark.add(c)
  for kid in kids[c]:
    if kid not in mark:
      _dfs(kid, kids, mark)

class Ontology():

  def _load_oboFile(self, oboFile, root):
    names={}
    def_text={}
    kids={}
    parents={}
    real_id = {}
    while True:
      line=oboFile.readline()
      if line == "":
        break
      tag, value = _get_tag_value(line)
      if tag == "":
        continue
      if tag == "id":
        concept_id = value
        parents[concept_id] = []
        kids[concept_id] = []
        names[concept_id] = []
        def_text[concept_id] = []
        real_id[concept_id] = concept_id

      if tag == "def":
        def_text[concept_id].append(value)
      if tag == "name" or tag == "synonym":
        names[concept_id].append(value)
      if tag == "alt_id":
        real_id[value] = concept_id
    
    oboFile.seek(0)
    while True:
      line=oboFile.readline()
      if line == "":
        break
      tag, value = _get_tag_value(line)
      if tag == "":
        continue
      if tag=="id":
        concept_id = value

      if tag=="is_a":
        parent_id = real_id[value]
        kids[parent_id].append(concept_id)
        parents[concept_id].append(parent_id)
    mark=set()
    _dfs(root, kids, mark)

    self.names = {c:names[c] for c in mark}
    self.def_text = {c:def_text[c] for c in mark if c in def_text}
    self.parents = {c:parents[c] for c in mark}
    self.kids = {c:kids[c] for c in mark}
    self.real_id = real_id
    for c in self.parents:
      self.parents[c]=[p for p in parents[c] if p in mark]

  def _update_ancestry_sparse(self, c):
    cid = self.concept2id[c]
    if cid in self.ancestor_weight:
      return self.ancestor_weight[cid].keys()

    self.ancestor_weight[cid] = {cid:1.0}

    num_parents = len(self.parents[c])

    for p in self.parents[c]:
      tmp_ancestors = self._update_ancestry_sparse(p)
      pid = self.concept2id[p]
      for ancestor in tmp_ancestors:
        if ancestor not in self.ancestor_weight[cid]:
          self.ancestor_weight[cid][ancestor] = 0.0 
        self.ancestor_weight[cid][ancestor] += (
                self.ancestor_weight[pid][ancestor]/num_parents)

    return self.ancestor_weight[cid].keys()



  def __init__(self, obo_address, root_id):
    if isinstance(obo_address, str):
      obo_file=open(obo_address)
    else:
      obo_file=obo_address
    self._load_oboFile(obo_file, root_id)
    self.concepts = [c for c in sorted(self.names.keys())]
    self.concept2id = dict(zip(self.concepts,range(len(self.concepts))))
    self.root_id = root_id

    self.name2conceptid = {}
    for c in self.concepts:
      for name in self.names[c]:
        normalized_name = name.strip().lower()
        self.name2conceptid[normalized_name] = self.concept2id[c]

    self.ancestor_weight = {}
    self.samples = []
    for c in self.concepts:
      self._update_ancestry_sparse(c)

    self.sparse_ancestors = []
    self.sparse_ancestors_values = []
    for cid in self.ancestor_weight:
      self.sparse_ancestors += [[cid, ancid]
          for ancid in self.ancestor_weight[cid]]
      self.sparse_ancestors_values += [self.ancestor_weight[cid][ancid]
          for ancid in self.ancestor_weight[cid]]

