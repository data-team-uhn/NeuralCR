import numpy as np
import re

from orangecontrib.bio.ontology import OBOOntology

def is_descendant_of(ont, itm, root):
  try:
    for t in ont.orange_model.super_terms(itm):
      if t.id == root:
        return True
  except:
    return False
  return False

def remove_from_list(lst, val):
  lst_copy = []
  for v in lst:
    if v != val:
      lst_copy.append(v)
  return lst_copy

class Ontology():
  def __init__(self, obo_address, root_id):
    self.orange_model = OBOOntology(obo_address)
    
    self.concepts = []
    self.names = {}
    self.parents = {}
    for t in self.orange_model.terms():
      if t.id is None:
        continue
      if t.name is None:
        continue
      if (not is_descendant_of(self, t.id, root_id)) and (t.id != root_id):
        continue
      
      self.concepts.append(t.id)
      self.names[t.id] = [t.name]
      for t_syn in t.synonyms:
        self.names[t.id].append(t_syn)
      self.parents[t.id] = []
      for rel_edge, rel_obj in t.related_objects():
        if rel_edge == 'is_a':
          #Only append() parents that are descendants of root
          if is_descendant_of(self, rel_obj, root_id) or rel_obj == root_id:
            self.parents[t.id].append(rel_obj)
    
    #Do a clean-up, removing parents that are not in the self.concepts list
    for c in self.concepts:
      remove_these = []
      for cp in self.parents[c]:
        if cp not in self.concepts:
          remove_these.append(cp)
      
      for itm in remove_these:
        self.parents[c] = remove_from_list(self.parents[c], itm)
    
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
  
  def to_dict(self):
    ret = {}
    ret['concepts'] = self.concepts
    ret['names'] = self.names
    ret['concept2id'] = self.concept2id
    ret['sparse_ancestors'] = self.sparse_ancestors
    ret['sparse_ancestors_values'] = self.sparse_ancestors_values
    ret['root_id'] = self.root_id
    ret['name2conceptid'] = self.name2conceptid
    return ret

