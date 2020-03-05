import tensorflow as tf
import numpy as np
import random
import json
import pickle 
import fasttext
import re
tf.enable_eager_execution()

def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

def tokenize(phrase):
  pattern = re.compile('[\W_]')
  tmp = pattern.sub(' ', phrase).lower().strip().split()
  return ["INT" if w.isdigit() else
      ("FLOAT" if is_number(w) else w) for w in tmp]

def phrase2vec(word_model, phrase_list, max_length):
  phrase_vec_list = []
  phrase_seq_lengths = []
  for phrase in phrase_list:
    tokens = tokenize(phrase)[:max_length-1]
    embedings = np.stack([word_model.get_word_vector(x) for x in tokens])
    pad = np.zeros([max_length-len(tokens),word_model.get_dimension()])
    phrase_vec_list.append(np.concatenate([embedings,pad]))
    phrase_seq_lengths.append(len(tokens))
  seq = np.stack(phrase_vec_list).astype(np.float32)
  seq_len = np.array(phrase_seq_lengths).astype(np.float32)
  return seq, seq_len 

class HierarchicalAggregate(tf.keras.layers.Layer):

  def __init__(self, n_concepts, sparse_ancestors, sparse_ancestors_values):
    super(HierarchicalAggregate, self).__init__()
    self.n_concepts = n_concepts
    self.ancestry_sparse_tensor = tf.sparse_reorder(tf.SparseTensor(
      indices = sparse_ancestors,
      values = sparse_ancestors_values,
      dense_shape=[self.n_concepts, self.n_concepts]))

  def build(self, input_shape):
    self.w = self.add_weight(
        'raw_embeddings',
        shape=(self.n_concepts, int(input_shape[-1])),
        initializer=tf.keras.initializers.RandomNormal(0, 0.01),
        trainable=True)
    self.b = self.add_weight(
        'bias',
        shape=(self.n_concepts,),
        initializer=tf.keras.initializers.RandomNormal(0, 0.01),
        trainable=True)

  def call(self, inputs):
    final_w = tf.transpose(
        tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, self.w))
    return tf.matmul(inputs, final_w) + self.b

class NCRCore(tf.keras.models.Sequential):

  def __init__(self, config, ont):
    super(NCRCore, self).__init__()
    n_concepts = len(ont.concepts) + 1

    self.add(tf.keras.layers.Conv1D(config.cl1, 1,
      activation=tf.keras.activations.elu,
      kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.1),
      bias_initializer=tf.keras.initializers.RandomNormal(0, 0.01)))

    # TODO: Add mask for seq len

    self.add(tf.keras.layers.Lambda(
        lambda z: tf.keras.backend.max(z, axis=1)))

    self.add(tf.keras.layers.Dense(config.cl2,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.1),
      bias_initializer=tf.keras.initializers.RandomNormal(0, 0.01)))

    if not config.no_l2norm:
      self.add(tf.keras.layers.Lambda(
          lambda z: tf.keras.backend.l2_normalize(z, axis=1)))

    if config.flat:
      self.add(tf.keras.layers.Dense(n_concepts,
        kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)))
    else:
      self.add(HierarchicalAggregate(
        n_concepts, ont.sparse_ancestors, ont.sparse_ancestors_values))

class NCR():
  def __init__(self, config, ont, word_model_file):
    self.config = config
    self.ont = ont
    print('Loading the fasttext model...')
    self.word_model = fasttext.load_model(word_model_file)

    print('Initializing NCR parameters...')
    self.ncr_cores = [NCRCore(config, ont) for i in range(config.n_ensembles)]

    inputs = tf.keras.Input(
        shape=(config.max_sequence_length, self.word_model.get_dimension()))
    outputs = [tf.keras.layers.Softmax()(ncr_core(inputs))
            for ncr_core in self.ncr_cores]
    if config.n_ensembles == 1:
      merged_outputs = outputs[0]
    else:
      merged_outputs = tf.keras.layers.Average()(outputs)
    self.ensembled_ncr =  tf.keras.Model(inputs=inputs, outputs=merged_outputs)

  @classmethod
  def loadfromfile(cls, param_dir, word_model_file):
    ont = pickle.load(open(param_dir+'/ont.pickle',"rb" )) 

    class Config(object):
      def __init__(self, d):
        self.__dict__ = d
    config = Config(json.load(open(param_dir+'/config.json', 'r')))

    model = cls(config, ont, word_model_file)
    model.ensembled_ncr.load_weights(param_dir+'/ncr_weights.h5')
    return model

  def save_weights(self, param_dir):
    self.ensembled_ncr.save_weights(
        param_dir+'/ncr_weights.h5', save_format='h5')

  def get_match(self, querry, count=1):
    batch_size = 512
    head = 0

    was_string = False
    if isinstance(querry, str):
      was_string = True
      querry = [querry]

    seq, seq_len = phrase2vec(
        self.word_model, querry, self.config.max_sequence_length)

    result_probs = []
    for head in range(0, len(querry), batch_size):
      querry_subset = seq[head:head+batch_size]
      result_probs.append(self.ensembled_ncr(querry_subset).numpy())
    res_querry = np.concatenate(result_probs)

    results=[]
    indecies_querry = np.argpartition(res_querry, -count, axis=-1)[:,-count:]
    for s in range(len(querry)):
      tmp_indecies_querry = indecies_querry[s,
          np.argsort(-res_querry[s,indecies_querry[s]])]
      tmp_res = []
      for i in tmp_indecies_querry:
        if i == len(self.ont.concepts):
          tmp_res.append(('None',res_querry[s,i]))
        else:
          tmp_res.append((self.ont.concepts[i],res_querry[s,i]))
        if len(tmp_res)>=count:
          break
      results.append(tmp_res)
    if was_string:
      return results[0]
    return results

  def annotate_text(self, text, threshold=0.8):
    pattern = re.compile('[\\\\/\r\n\t-]')
    text_replaced = pattern.sub(' ', text)
    chunks_large = re.split('[^a-zA-Z ]',text_replaced)
    candidates = []
    candidates_info = []
    total_chars=0
    for c,chunk in enumerate(chunks_large):
      tokens = chunk.split(" ")
      chunk_chars = 0
      for i,w in enumerate(tokens):
        phrase = ""
        for r in range(7):
          if i+r >= len(tokens) or len(tokens[i+r])==0:
            break
          if r>0:
            phrase += " " + tokens[i+r]
          else:
            phrase = tokens[i+r]
          cand_phrase = phrase
          if len(cand_phrase) > 0:
            candidates.append(cand_phrase)
            location = total_chars+chunk_chars
            candidates_info.append((location, location+len(phrase), c))
        chunk_chars += len(w)+1
      total_chars += len(chunk)+1
    matches = [x[0] for x in self.get_match(candidates, 1)]
    filtered = {}

    for i in range(len(candidates)): #TODO
      if (matches[i][0]!=self.ont.root_id and matches[i][0]!="None" and
          matches[i][1]>threshold):
        if candidates_info[i][2] not in filtered:
          filtered[candidates_info[i][2]] = []
        filtered[candidates_info[i][2]].append((
          candidates_info[i][0],
          candidates_info[i][1],
          matches[i][0],
          matches[i][1]))

    final = [] 
    for c in filtered:
      tmp_final = []
      for x in filtered[c]:
        bad = False
        for y in filtered[c]:
          if (x[0]<=y[0] and x[1]>=y[1] and x[2]==y[2]
              and (x is not y) and x[3]<y[3]):
            bad=True
            break
        if not bad:
          tmp_final.append(x)
      cands = sorted(tmp_final, key= lambda x:x[0]-x[1])
      tmp_final = []
      for x in cands:
        conflict = False
        for y in tmp_final:
          if x[1]>y[0] and x[0]<y[1]:
            conflict = True
            break
        if not conflict:
          tmp_final.append(x)
      final+=tmp_final
    return final

