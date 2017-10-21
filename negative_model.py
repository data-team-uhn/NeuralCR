import tensorflow as tf
import numpy as np
import random

class Config:
    include_negs = True
    batch_size = 256
    cl1 = 1024
    cl2 = 1024
    cl3 = 512

    max_sequence_length = 50
    lr = 1.0/512 

    word_embed_size = 100
    @staticmethod
    def update_with_reader(ont):
        Config.concepts_size = len(ont.concepts)+1

def linear(name, x, shape, initializer=tf.random_normal_initializer(stddev=0.1)):
    w = tf.get_variable(name+"W", shape = shape, initializer = initializer)
    b = tf.get_variable(name+"B", shape = shape[1], initializer = tf.random_normal_initializer(stddev=0.01))
    return tf.matmul(x,w) + b

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def tokenize(phrase):
    tmp = phrase.lower().replace(',',' , ').replace('-',' ').replace(';', ' ; ').replace('/', ' / ').replace('(', ' ( ').replace(')', ' ) ').replace('.', ' . ').strip().split()
    return ["INT" if w.isdigit() else ("FLOAT" if is_number(w) else w) for w in tmp]

def linear_sparse(name, x, shape):
#    w = tf.get_variable(name+"W", shape, initializer=tf.contrib.layers.xavier_initializer())
    w = tf.get_variable(name+"W", shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))
#    w = weight_variable(name+"W", shape)
#    b = tf.get_variable(name+"B", shape = shape[1], initializer = tf.random_normal_initializer(stddev=0.1))
#    b = weight_variable(name+"B", [shape[1]])
    return tf.sparse_tensor_dense_matmul(x,w) #+ b

def euclid_dis_cartesian(v, u):
    return tf.reduce_sum(v*v, 1, keep_dims=True) + tf.expand_dims(tf.reduce_sum(u*u, 1), 0) - 2 * tf.matmul(v,u, transpose_b=True) 


class NegModel():
    def phrase2vec(self, phrase_list, max_length):
        phrase_vec_list = []
        phrase_seq_lengths = []
        for phrase in phrase_list:
            tokens = tokenize(phrase)[:max_length-1]
            # TODO get the embedding size
            phrase_vec_list.append([self.word_model[tokens[i]] if i<len(tokens) else [0]*self.word_model.dim for i in range(max_length)])
            phrase_seq_lengths.append(len(tokens))
        return np.array(phrase_vec_list), np.array(phrase_seq_lengths)

    def process_text(self, text, threshold=0.5):
#        chunks_large = text.replace("\r"," ").replace("\n"," ").replace("\t", " ").replace(",","|").replace(";","|").replace(".","|").replace("-","|").split("|")
        chunks_large = text.replace("\r"," ").replace("\n"," ").replace("\t", " ").replace(",","|").replace(";","|").replace(".","|").split("|")
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
                    #cand_phrase = phrase.strip(',/;-.').strip()
                    cand_phrase = phrase
                    if len(cand_phrase) > 0:
                        candidates.append(cand_phrase)
                        location = total_chars+chunk_chars
                        candidates_info.append((location, location+len(phrase), c))
                chunk_chars += len(w)+1
            total_chars += len(chunk)+1
        matches = [x[0] for x in self.get_hp_id(candidates, 1)]
        filtered = {}


        for i in range(len(candidates)):
            if matches[i][0]!='HP:0000118' and matches[i][0]!="None" and matches[i][2]>threshold:
                if candidates_info[i][2] not in filtered:
                    filtered[candidates_info[i][2]] = []
                filtered[candidates_info[i][2]].append((candidates_info[i][0], candidates_info[i][1], matches[i][0], matches[i][1]))

        final = [] 
        for c in filtered:
            tmp_final = []
            for x in filtered[c]:
                bad = False
                for y in filtered[c]:
                    if x[0]<=y[0] and x[1]>=y[1] and x[2]==y[2] and (x is not y): #(m2[1]-m2[0]<best_smaller[1]-best_smaller[0]):
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



        '''
        for c in filtered:
            tmp_final = []
            cands = sorted(filtered[c], key= lambda x:x[0]-x[1])
            for m in cands:
                conflict = False
                for m2 in tmp_final:
                    if m[1]>m2[0] and m[0]<m2[1]:
                        conflict = True
                        break
                if conflict:
                    continue
                best_smaller = m
                for m2 in cands:
                    if m[0]<=m2[0] and m[1]>=m2[1] and m[2]==m2[2] and (m2[1]-m2[0]<best_smaller[1]-best_smaller[0]):
                        best_smaller = m2
                tmp_final.append(best_smaller)
            final+=tmp_final
        '''
        return final


    '''
    def get_score(self, embedding):
        last_layer_w =  linear_sparse('last_layerW', self.ancestry_sparse_tensor, [self.config.concepts_size, self.config.cl2])
        last_layer_b = tf.get_variable('last_layer'+"B", shape = [self.config.concepts_size], initializer = tf.random_normal_initializer(stddev=0.001))

        score_layer = tf.matmul(embedding, tf.transpose(last_layer_w))  + last_layer_b
        return score_layer
    '''


    #############################
    ##### Creates the model #####
    #############################
    def __init__(self, config, ont, word_model):
        print "Creating the model graph" 
        tf.reset_default_graph()
        self.ont = ont
        self.word_model = word_model
        config.update_with_reader(self.ont)
        self.config = config

        ### Inputs ###
        self.label = tf.placeholder(tf.int32, shape=[None])

        self.seq = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, config.word_embed_size])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])
	self.lr = tf.Variable(config.lr, trainable=False)
        self.is_training = tf.placeholder(tf.bool)
        
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
       
        cell_fw = tf.nn.rnn_cell.GRUCell(100, kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        cell_bw = tf.nn.rnn_cell.GRUCell(100, kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        bidirectional_dynamic_rnn(
        

	self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        print "initializing"
	self.sess.run(tf.global_variables_initializer())
        print "initialized"


    def save_params(self, repdir='.'):
        tf.train.Saver().save(self.sess, (repdir+'/params.ckpt').replace('//','/'))

    def load_params(self, repdir='.'):
        tf.train.Saver().restore(self.sess, (repdir+'/params.ckpt').replace('//','/'))

    def init_training(self, neg_samples=None):
        raw_samples = []
        labels = []
        for c in self.ont.concepts:
            for name in self.ont.names[c]:
                raw_samples.append(name)
                labels.append(self.ont.concept2id[c]) 
        if neg_samples!=None:
            weights = np.array([1.0/len(raw_samples)]*(self.config.concepts_size-1)+[1.0/len(neg_samples)])*(len(neg_samples)+len(raw_samples))/2.0
            self.sess.run(tf.assign(self.class_weights, weights ))

            none_id = len(self.ont.concepts)
            raw_samples+=neg_samples
            labels += [none_id]*len(neg_samples)

        self.training_samples = {}
        self.training_samples['seq'], self.training_samples['seq_len'] = self.phrase2vec(raw_samples, self.config.max_sequence_length)
        self.training_samples['label'] = np.array(labels)

    def train_epoch(self, verbose=True):
	ct = 0
	report_loss = 0
	total_loss = 0
        report_len = 20
        head = 0
        training_size = self.training_samples['seq'].shape[0]
        shuffled_indecies = range(training_size)
        random.shuffle(shuffled_indecies)
	while head < training_size:
            ending = min(training_size, head + self.config.batch_size)
            batch = {}
            for cat in self.training_samples:
                batch[cat] = self.training_samples[cat][shuffled_indecies[head:ending]]
            head += self.config.batch_size
            batch_feed = {self.seq:batch['seq'],\
                    self.seq_len:batch['seq_len'],\
                    self.label:batch['label'], 
                    self.is_training:True} 
            _ , batch_loss = self.sess.run([self.train_step, self.loss], feed_dict = batch_feed)
            report_loss += batch_loss
            total_loss += batch_loss
            if verbose and ct % report_len == report_len-1:
                print "Step =", ct+1, "\tLoss =", report_loss/report_len
                #print np.median(self.sess.run(tf.norm(self.final_w, axis=-1)))
                #print np.median(self.sess.run(tf.norm(self.w, axis=-1)))
                #print np.median(self.sess.run(tf.reduce_sum(tf.abs(self.final_w), axis=-1)))
                #sys.stdout.flush()
                report_loss = 0
            ct += 1

        return total_loss/ct

    def get_probs(self, querry):
        seq, seq_len = self.phrase2vec(querry, self.config.max_sequence_length)

        querry_dict = {self.seq : seq, self.seq_len: seq_len, self.is_training:False}
#        res_querry = self.sess.run(self.score_layer, feed_dict = querry_dict)
        res_querry = self.sess.run([self.pred, self.agg_pred], feed_dict = querry_dict)
        return res_querry

    def get_hp_id(self, querry, count=1):
        #if self.anchors_set:
        #    return self.get_hp_id_from_anchor(querry, count)
        batch_size = 512
        head = 0

        while head < len(querry):
            querry_subset = querry[head:min(head+batch_size, len(querry))]
            res_tmp, agg_pred_tmp = self.get_probs(querry_subset)
            if head == 0:
                res_querry = res_tmp #self.get_probs(querry_subset)
                agg_pred = agg_pred_tmp #self.get_probs(querry_subset)
            else:
                res_querry = np.concatenate((res_querry, res_tmp))
                agg_pred = np.concatenate((agg_pred, agg_pred_tmp))

            head += batch_size

        #res_querry = self.get_probs(querry)

        results=[]
        for s in range(len(querry)):
            indecies_querry = np.argsort(-res_querry[s,:])

            tmp_res = []
            for i in indecies_querry:
                '''
                print i
                '''
                if i == len(self.ont.concepts):
                    tmp_res.append(('None',res_querry[s,i]))
                else:
                    tmp_res.append((self.ont.concepts[i],res_querry[s,i], agg_pred[s, 1]))
                if len(tmp_res)>=count:
                        break
            results.append(tmp_res)

        return results

