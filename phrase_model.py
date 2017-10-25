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


class NCRModel():
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
            #tokens = tokenize(chunk)
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
            if matches[i][0]!='HP:0000118' and matches[i][0]!="None" and matches[i][1]>threshold:
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
        self.class_weights = tf.Variable(tf.ones([config.concepts_size]), False)

        self.seq = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, config.word_embed_size])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])
	self.lr = tf.Variable(config.lr, trainable=False)
        self.is_training = tf.placeholder(tf.bool)

        self.ancestry_sparse_tensor = tf.sparse_reorder(tf.SparseTensor(indices = ont.sparse_ancestrs, values = [1.0]*len(ont.sparse_ancestrs), dense_shape=[config.concepts_size, config.concepts_size]))

        #######################
        ## Phrase embeddings ##
        #######################

        '''
        filters = [self.config.cl1]
        #filters = [self.config.cl1, self.config.cl1/4, self.config.cl1/4]
        grams = [tf.layers.conv1d(self.seq, filters[i], i+1, activation=tf.nn.relu,\
                kernel_initializer=tf.random_normal_initializer(0.0,0.1), bias_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=True) for i in range(len(filters))]

        grams = [grams[i]*tf.expand_dims(tf.sequence_mask(tf.maximum(self.seq_len-i,0), config.max_sequence_length-i, dtype=tf.float32),axis=-1) for i in range(len(filters))]
                #kernel_initializer=tf.random_normal_initializer(0.0,0.1), use_bias=True) for i in range(len(filters))]

        #filters2 = [self.config.cl2]
        #grams = [tf.layers.conv1d(grams[i], filters2[i], i+1, activation=tf.nn.elu,\
        #        kernel_initializer=tf.random_normal_initializer(0.0,0.1), use_bias=True) for i in range(len(filters2))]

        grams_pooled = [tf.reduce_max(gram, [1]) for gram in grams]
        layer1_pooled = tf.concat(grams_pooled, axis=-1)
        layer2 = tf.layers.dense(layer1_pooled, self.config.cl2, tf.nn.relu,\
                kernel_initializer=tf.random_normal_initializer(0.0,0.1))  
        self.seq_embedding = tf.nn.l2_normalize(layer2  , dim=1)
        #self.seq_embedding = tf.layers.dropout(self.seq_embedding, rate=0.3, training=self.is_training)



        '''

        '''
        layer1 = tf.layers.conv1d(self.seq, self.config.cl1, 1, activation=tf.nn.tanh,\
                kernel_initializer=tf.contrib.layers.xavier_initializer(),\
                bias_initializer=tf.random_normal_initializer(0.0,0.1), use_bias=True)
        '''


        #########
        layer1 = tf.layers.conv1d(self.seq, self.config.cl1, 1, activation=tf.nn.elu,\
                kernel_initializer=tf.random_normal_initializer(0.0,0.1),\
                bias_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=True)

        layer2 = tf.layers.dense(tf.reduce_max(layer1, [1]), self.config.cl2, activation=tf.nn.relu,\
        #layer2 = tf.layers.dense(tf.reduce_max(layer1, [1]), self.config.cl2, activation=tf.nn.relu,\
                kernel_initializer=tf.random_normal_initializer(0.0,stddev=0.1),
                bias_initializer=tf.random_normal_initializer(0.0,stddev=0.01), use_bias=True)

        '''
        layer3 = tf.layers.dense(layer2, self.config.cl3, activation=tf.nn.tanh,\
                kernel_initializer=tf.random_normal_initializer(0.0,stddev=0.1),
                bias_initializer=tf.random_normal_initializer(0.0,stddev=0.01), use_bias=True)
        '''
        #########
        '''
        layer2 = tf.layers.dense(tf.reduce_max(layer1, [1]), self.config.cl1, tf.nn.tanh,\
                kernel_initializer=tf.contrib.layers.xavier_initializer(),\
                bias_initializer=tf.random_normal_initializer(0.0,0.1), use_bias=True)

        layer3 = tf.layers.dense(layer2, self.config.cl2,\
                kernel_initializer=tf.random_normal_initializer(0.0,0.1))  
        '''

        #self.seq_embedding = layer3
        #cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(128) for _ in range(1)])
        #outputs, state = tf.nn.dynamic_rnn(cell, self.seq, self.seq_len, dtype=tf.float32)
        
        #self.seq_embedding = state
#        self.seq_embedding = layer3
        self.seq_embedding = tf.nn.l2_normalize(layer2  , dim=1)

        '''
        filters1 = tf.get_variable('conv1', [1, self.config.word_embed_size, self.config.cl1], tf.float32, initializer = tf.random_normal_initializer(stddev=0.1))
        conv1_b = tf.get_variable('0conv1_b', initializer=tf.random_normal_initializer(stddev=0.01), shape=self.config.cl1)
        layer1 = tf.nn.elu(tf.nn.conv1d(self.seq, filters1, 1, padding='SAME') + conv1_b)
        '''

        #self.seq_embedding = tf.nn.l2_normalize(tf.nn.relu(linear('lassst', tf.reduce_max(layer1, [1]), [self.config.cl2, self.config.cl2]))  , dim=1)

        ########################
        ## Concept embeddings ##
        ########################
        self.embeddings = tf.get_variable("embeddings", shape = [self.config.concepts_size, self.config.cl2], initializer = tf.random_normal_initializer(stddev=0.1))
        #self.embeddings = tf.nn.l2_normalize(self.embeddings, dim=1)
        self.aggregated_embeddings = tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, self.embeddings) 
        aggregated_w = self.aggregated_embeddings
        #aggregated_w = tf.nn.l2_normalize(self.aggregated_embeddings, dim=1)
#        aggregated_w = self.embeddings
#        aggregated_w = tf.nn.tanh(self.aggregated_embeddings)
        '''
        aggregated_w = tf.layers.dense(aggregated_w, self.config.cl2,\
                kernel_initializer=tf.random_normal_initializer(0.0,stddev=0.1),
                bias_initializer=tf.random_normal_initializer(0.0,stddev=0.01), use_bias=True)
        '''

        last_layer_b = tf.get_variable('last_layer_bias', shape = [self.config.concepts_size], initializer = tf.random_normal_initializer(stddev=0.001))

#        direct_score_layer = tf.layers.dense(self.seq_embedding, self.config.concepts_size,\
#                kernel_initializer=tf.random_normal_initializer(0.0,0.1))
#        aggregated_score_layer = -euclid_dis_cartesian(self.seq_embedding, aggregated_w)
        aggregated_score_layer = tf.matmul(self.seq_embedding, tf.transpose(aggregated_w)) + last_layer_b
        
        self.score_layer = aggregated_score_layer
#        self.score_layer = direct_score_layer
        ########################
        ########################
        ########################

        #self.seq_embedding = tf.layers.dropout(self.seq_embedding, rate=0.3, training=self.is_training)
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.label, self.score_layer)) 


#        self.seq_embedding = self.apply_meanpool(self.seq, self.seq_len)
#        self.score_layer = self.get_score(self.seq_embedding)
        label_one_hot = tf.one_hot(self.label, config.concepts_size)

    
        self.pred = tf.nn.softmax(self.score_layer)
        self.agg_pred, _ =  tf.nn.top_k(tf.transpose(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.ancestry_sparse_tensor), tf.transpose(self.pred))), 2)

#        self.pred = self.score_layer/tf.reduce_sum(self.score_layer,-1)
#        self.loss = -tf.reduce_mean(tf.reduce_sum(label_one_hot * self.score_layer,-1)) 
#        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.label, self.score_layer, tf.gather(self.class_weights, self.label)))

        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.apply_gradients(zip(gradients, variables))
            #self.train_step = optimizer.apply_gradients(capped_gvs)
        '''
        
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        

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

