import argparse
import ncrmodel 
import numpy as np
import os
from onto import Ontology
import json

#Necessary so that it works both locally and also on HPF
try:
  import fasttext as fastText
except ImportError:
  import fastText

import pickle 
import tensorflow as tf
import accuracy
import annotate_text
import eval
import tempfile
import shutil
import random
#tf.enable_eager_execution() #For HPF

def save_ont_and_args(ont, args, param_dir):
  pickle.dump(ont, open(param_dir+'/ont.pickle',"wb" )) 
  with open(param_dir+'/config.json', 'w') as fp:
    json.dump(vars(args),fp)

def sample_negatives_from_file(file_addr, count):
  max_text_size = 10*1000*1000
  with open(file_addr, errors='replace') as f:
    text = f.read()[:max_text_size]

  tokens = ncrmodel.tokenize(text)
  indecies = np.random.choice(len(tokens), count)
  lengths = np.random.randint(1, 10, count)
  negative_samples = [' '.join(tokens[indecies[i]:indecies[i]+lengths[i]])
      for i in range(count)]
  return negative_samples

def main():
  parser = argparse.ArgumentParser(description='Hello!')
  parser.add_argument('--obofile', help="address to the ontology .obo file")
  parser.add_argument('--oboroot', help="the concept in the ontology to be used as root (only this concept and its descendants will be used)")
  parser.add_argument('--fasttext', help="address to the fasttext word vector file")
  parser.add_argument('--neg_file', help="address to the negative corpus", default="")
  parser.add_argument('--output', help="address to the directroy where the trained model will be stored")
  parser.add_argument('--output_without_early_stopping', help="address to the directroy where the trained model will be stored, without considering early stopping")
  parser.add_argument('--phrase_val', help="address to the file containing labeled phrases for validation")
  parser.add_argument('--flat', action="store_true", help="whether utilizing the concepts' hierarchy structure")
  parser.add_argument('--no_l2norm', action="store_true")
  parser.add_argument('--no_negs', action="store_true")
  parser.add_argument('--verbose', action="store_true")
  parser.add_argument('--cl1', type=int, help="cl1", default=1024)
  parser.add_argument('--cl2', type=int, help="cl2", default=1024)
  parser.add_argument('--lr', type=float, help="learning rate", default=1/512)
  parser.add_argument('--batch_size', type=int, help="batch size", default=256)
  parser.add_argument('--max_sequence_length', type=int, help="max sequence length", default=50)
  parser.add_argument('--epochs', type=int, help="number of epochs", default=80)
  parser.add_argument('--n_ensembles', type=int, help="number of ensembles", default=10)
  parser.add_argument('--num_negs', type=int, help="number of negative samples to use", default=10000)
  parser.add_argument('--validation_rate', type=int, help="number of epochs per validation", default=5)
  parser.add_argument('--sentence_val_input_dir', help="address to the directroy where the validation text files are stored")
  parser.add_argument('--sentence_val_label_dir', help="address to the directroy where the validation labels are stored")
  parser.add_argument('--snomed2icd')
  parser.add_argument('--eval_mimic', action="store_true")
  args = parser.parse_args()
  main_train(args)

#Here to support the remaining code which expects args to have come from parser.parse_args()
class MainTrainArgClass:
  def __init__(self,
   obofile=None,
   oboroot=None,
   fasttext=None,
   neg_file="",
   output=None,
   output_without_early_stopping=None,
   phrase_val=None,
   flat=None,
   no_l2norm=None,
   no_negs=None,
   verbose=None,
   cl1=1024,
   cl2=1024,
   lr=1/512,
   batch_size=256,
   max_sequence_length=50,
   epochs=80,
   n_ensembles=10,
   num_negs=10000,
   validation_rate=5,
   sentence_val_input_dir=None,
   sentence_val_label_dir=None,
   snomed2icd=None,
   eval_mimic=None):
    self.obofile = obofile
    self.oboroot = oboroot
    self.fasttext = fasttext
    self.neg_file = neg_file
    self.output = output
    self.output_without_early_stopping = output_without_early_stopping
    self.phrase_val = phrase_val
    self.flat = flat
    self.no_l2norm = no_l2norm
    self.no_negs = no_negs
    self.verbose = verbose
    self.cl1 = cl1
    self.cl2 = cl2
    self.lr = lr
    self.batch_size = batch_size
    self.max_sequence_length = max_sequence_length
    self.epochs = epochs
    self.n_ensembles = n_ensembles
    self.num_negs = num_negs
    self.validation_rate = validation_rate
    self.sentence_val_input_dir = sentence_val_input_dir
    self.sentence_val_label_dir = sentence_val_label_dir
    self.snomed2icd = snomed2icd
    self.eval_mimic = eval_mimic
  
  def to_json(self):
    ret = {}
    ret['obofile'] = self.obofile
    ret['oboroot'] = self.oboroot
    ret['fasttext'] = self.fasttext
    ret['neg_file'] = self.neg_file
    ret['output'] = self.output
    ret['output_without_early_stopping'] = self.output_without_early_stopping
    ret['phrase_val'] = self.phrase_val
    ret['flat'] = self.flat
    ret['no_l2norm'] = self.no_l2norm
    ret['no_negs'] = self.no_negs
    ret['verbose'] = self.verbose
    ret['cl1'] = self.cl1
    ret['cl2'] = self.cl2
    ret['lr'] = self.lr
    ret['batch_size'] = self.batch_size
    ret['max_sequence_length'] = self.max_sequence_length
    ret['epochs'] = self.epochs
    ret['n_ensembles'] = self.n_ensembles
    ret['num_negs'] = self.num_negs
    ret['validation_rate'] = self.validation_rate
    ret['sentence_val_input_dir'] = self.sentence_val_input_dir
    ret['sentence_val_label_dir'] = self.sentence_val_label_dir
    ret['snomed2icd'] = self.snomed2icd
    ret['eval_mimic'] = self.eval_mimic
    return json.dumps(ret)

def main_train(training_args):
  args = MainTrainArgClass(training_args.obofile,
    training_args.oboroot,
    training_args.fasttext,
    training_args.neg_file,
    training_args.output,
    training_args.output_without_early_stopping,
    training_args.phrase_val,
    training_args.flat,
    training_args.no_l2norm,
    training_args.no_negs,
    training_args.verbose,
    training_args.cl1,
    training_args.cl2,
    training_args.lr,
    training_args.batch_size,
    training_args.max_sequence_length,
    training_args.epochs,
    training_args.n_ensembles,
    training_args.num_negs,
    training_args.validation_rate,
    training_args.sentence_val_input_dir,
    training_args.sentence_val_label_dir,
    training_args.snomed2icd,
    training_args.eval_mimic)
  
  print('Loading the ontology...')
  ont = Ontology(args.obofile,args.oboroot)

  model = ncrmodel.NCR(args, ont, args.fasttext)
  if (not args.no_negs) and args.neg_file != "":
    negative_samples = sample_negatives_from_file(args.neg_file, args.num_negs)

  raw_data = []
  labels = []
  for c in ont.concepts:
    for name in ont.names[c]:
      raw_data.append(name)
      labels.append(ont.concept2id[c]) 
  if negative_samples!=None:
    none_id = len(ont.concepts)
    raw_data+=negative_samples
    labels += [none_id]*len(negative_samples)
  training_data = {}
  training_data['seq'], training_data['seq_len'] = ncrmodel.phrase2vec(
      model.word_model, raw_data, args.max_sequence_length)
  training_data['label'] = np.array(labels).astype(np.int32)
  training_data_size = training_data['seq'].shape[0]

  optimizers = [tf.train.AdamOptimizer(learning_rate=args.lr)
      for i in range(args.n_ensembles)]

  if args.phrase_val != None: 
    samples = accuracy.prepare_phrase_samples(model.ont, args.phrase_val, True)


  if args.snomed2icd != None:
    with open(args.snomed2icd, 'r') as fp:
      snomed2icd = json.load(fp)

  best_loss = -1.0
  param_dir = args.output
  best_result = -1.0
  if not os.path.exists(param_dir):
      os.makedirs(param_dir)
  #'''
  if args.sentence_val_input_dir != None:
    tmp_dirpath = tempfile.mkdtemp()

  report_len = 20
  for epoch in range(args.epochs):
    epoch_loss = 0
    epoch_ct = 0
    print("Epoch :: "+str(epoch))
    for ens_i, ncr_core in enumerate(model.ncr_cores):
      ct = 0
      report_loss = 0
      shuffled_indecies = list(range(training_data_size))
      random.shuffle(shuffled_indecies)
      for head in range(0, training_data_size, args.batch_size):
        batch_indecies = shuffled_indecies[
            head:head+args.batch_size]
        batch = {}
        for cat in training_data:
          batch[cat] = training_data[cat][batch_indecies]

        with tf.GradientTape() as tape:
          logits = ncr_core(batch['seq'])
          loss = tf.reduce_sum(
              tf.losses.sparse_softmax_cross_entropy(batch['label'], logits))
        grads = tape.gradient(loss, ncr_core.trainable_weights)
        optimizers[ens_i].apply_gradients(zip(grads, ncr_core.trainable_weights))

        report_loss += loss.numpy()
        epoch_loss += loss.numpy()
        if args.verbose and ct % report_len == report_len-1:
          print("Step = "+str(ct+1)+"\tLoss ="+str(report_loss/report_len))
          report_loss = 0
        ct += 1
        epoch_ct += 1
    print("epoch loss:" + str(epoch_loss/epoch_ct))
    if args.sentence_val_input_dir != None and (epoch==args.epochs-1 or (epoch%args.validation_rate==0 and epoch>min(args.epochs//2, 30))): 
        sent_input_stream = annotate_text.DirInputStream(args.sentence_val_input_dir)
        sent_output_stream = annotate_text.DirOutputStream(tmp_dirpath)
        annotate_text.annotate_stream(model, 0.8, sent_input_stream, sent_output_stream)
        file_list = os.listdir(tmp_dirpath)
        if args.eval_mimic:
            results = eval.eval_mimic(args.sentence_val_label_dir, tmp_dirpath, file_list, ont, snomed2icd, column=2)#, args.comp_dir)
        else:
            results = eval.eval(args.sentence_val_label_dir, tmp_dirpath, file_list, ont, column=2)#, args.comp_dir)
        if results['micro']['fmeasure']>best_result:
            best_result = results['micro']['fmeasure']
            model.save_weights(param_dir)
        print(results['micro']['fmeasure'])

    if args.phrase_val != None and (epoch%5==0 or epoch==args.epochs-1) and epoch>=0: 
        res = model.get_match(list(samples.keys()), 1)
        missed = [x for i,x in enumerate(samples) if samples[x] not in [r[0] for r in res[i]]]
        print("R@1: "+ str((len(samples)-len(missed))/len(samples)))

        res = model.get_match(list(samples.keys()), 5)
        missed = [x for i,x in enumerate(samples) if samples[x] not in [r[0] for r in res[i]]]
        print("R@5: "+ str((len(samples)-len(missed))/len(samples)))
    if epoch%5==0 and epoch>0: 
        for x in model.get_match('blood examination', 5):
            print(x[0], (ont.names[x[0]] if x[0]!='None' else x[0]), x[1])

  
  save_ont_and_args(ont, args, param_dir)
  if args.sentence_val_input_dir == None:
    model.save_weights(param_dir)
  else:
    shutil.rmtree(tmp_dirpath)
    os.makedirs(args.output_without_early_stopping)
    if args.output_without_early_stopping!=None:
      model.save_weights(args.output_without_early_stopping)
      save_ont_and_args(ont, args, args.output_without_early_stopping)
  
  print("Finished main_train()")

if __name__ == "__main__":
    main()
