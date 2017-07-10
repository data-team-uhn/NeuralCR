import tensorflow as tf
import numpy as np
import random


def weight_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def linear(name, x, shape, phase=0):
#    w = tf.get_variable(name+"W", shape, initializer=tf.contrib.layers.xavier_initializer())
    w = tf.get_variable(name+"W", shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))
#    w = weight_variable(name+"W", shape)
    b = tf.get_variable(name+"B", shape = shape[1], initializer = tf.random_normal_initializer(stddev=0.01))
#    b = weight_variable(name+"B", [shape[1]])
    return tf.matmul(x,w) + b

def linear_batch_norm(name, x, shape, phase):
#    w = tf.get_variable(name+"W", shape, initializer=tf.contrib.layers.xavier_initializer())
    w = tf.get_variable(name+"W", shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))
#    w = weight_variable(name+"W", shape)
    b = tf.get_variable(name+"B", shape = shape[1], initializer = tf.random_normal_initializer(stddev=0.1))
#    b = weight_variable(name+"B", [shape[1]])
    return tf.contrib.layers.batch_norm(tf.matmul(x,w) + b, center=True, scale=True, updates_collections=None,
                                          is_training=phase)


def linear_sparse(name, x, shape):
#    w = tf.get_variable(name+"W", shape, initializer=tf.contrib.layers.xavier_initializer())
    w = tf.get_variable(name+"W", shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))
#    w = weight_variable(name+"W", shape)
#    b = tf.get_variable(name+"B", shape = shape[1], initializer = tf.random_normal_initializer(stddev=0.1))
#    b = weight_variable(name+"B", [shape[1]])
    return tf.sparse_tensor_dense_matmul(x,w) #+ b


def embedding_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def tokenize(phrase):
	tmp = phrase.lower().replace(',',' , ').replace('-',' ').replace(';', ' ; ').replace('/', ' / ').replace('(', ' ( ').replace(')', ' ) ').replace('.', ' . ').strip().split()
	return ["INT" if w.isdigit() else ("FLOAT" if is_number(w) else w) for w in tmp]


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
    '''
	def phenotips(self, phrases, count=1):
		results = []
		for phrase in phrases:
			resp = requests.get('https://phenotips.org/get/PhenoTips/SolrService?vocabulary=hpo&q='+phrase.replace(" ","+")).json()
			ans = [(self.rd.real_id[str(x[u'id'])],x[u'score']) if str(x[u'id']) in self.ant.rd.real_id else (x[u'id'],x[u'score']) for x in resp['rows'][:count]]
			results.append(ans)
		return results
    '''
    def apply_meanpool(self, seq, seq_length):
        filters1 = tf.get_variable('conv1', [1, self.config.word_embed_size, self.config.cl1], tf.float32, initializer = tf.random_normal_initializer(stddev=0.1))
        conv1_b = tf.get_variable('0conv1_b', initializer=tf.random_normal_initializer(stddev=0.01), shape=self.config.cl1)
        seq_dp = tf.nn.dropout(seq, self.keep_prob)
        layer1 = tf.nn.elu(tf.nn.conv1d(seq_dp, filters1, 1, padding='SAME') + conv1_b)

        filters2 = tf.get_variable('conv2', [1, self.config.cl1, self.config.cl2], tf.float32, initializer = tf.random_normal_initializer(stddev=0.1))
        conv2_b = tf.get_variable('0conv2_b', initializer=tf.random_normal_initializer(stddev=0.01), shape=self.config.cl2)
        layer1_dp = layer1 #tf.nn.dropout(layer1, self.keep_prob)
        self.layer2 = tf.nn.elu(tf.nn.conv1d(layer1_dp, filters2, 1, padding='SAME') + conv2_b)

        with tf.variable_scope('top_cnn'):
            ret = tf.nn.l2_normalize(tf.nn.relu(linear('lassst', tf.reduce_max(self.layer2, [1]), [self.config.cl2, self.config.cl2]))  , dim=1)
        return ret
    def get_HPO_embedding(self, indices=None):
        embedding = self.HPO_embedding
        if indices is not None:
            embedding = tf.gather(self.HPO_embedding, indices)
            return embedding #tf.maximum(0.0, embedding)

    def annotate_text(self, text, threshold=0.2): ##TODO list to str
        chunks_large = text.replace("\r","|").replace("\n","|").replace("\t", " ").replace(",","|").replace(";","|").replace(".","|").split("|")
        inp = self.rd.create_test_sample(chunks_large)
        bsize = 256
        total = inp['seq'].shape[0]
        head = 0
        ret = []
        while head<total:
            small_inp = {}
            for c in inp:
                small_inp[c]=inp[c][head:min(head+bsize,total)]
            ret+= self.extract_from_sentence(small_inp, threshold)
            head+=bsize
        return ret



    def extract_from_sentence(self, inp, threshold): ##TODO list to str
        querry_dict = {self.seq : inp['seq'], self.seq_len: inp['seq_len'], self.keep_prob:1.0}
        res_querry = self.sess.run(self.wins, feed_dict = querry_dict)
        hp_index = [np.argmax(x, axis=-1) for x in res_querry]
        score = [np.max(x, axis=-1) for x in res_querry]
        candidates = [np.argwhere((score[i]>threshold)  & (hp_index[i]!=self.config.concepts_size-1)& (hp_index[i]!=self.rd.concept2id['HP:0000118'])) for i in range(len(score))]
        filtered = []
        for i in range(len(score)):
            for indx in candidates[i]:
                while indx[0]>=len(filtered):
                    filtered.append([])
                filtered[indx[0]].append((indx[1], indx[1]+i, self.rd.concepts[hp_index[i][tuple(indx)]], score[i][tuple(indx)]))
        final = []
        for clist in filtered:
            tmp_final = []
            cands = sorted(clist, key= lambda x:x[0]-x[1])
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

        return final

    def process_text(self, text, threshold=0.5):
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
        #print "---->>>>"
        #print matches
        #print " "
        for i in range(len(candidates)):
            if matches[i][0]!='HP:0000118' and matches[i][0]!="HP:None" and matches[i][1]>threshold:
                if candidates_info[i][2] not in filtered:
                    filtered[candidates_info[i][2]] = []
                filtered[candidates_info[i][2]].append((candidates_info[i][0], candidates_info[i][1], matches[i][0], matches[i][1]))

        #print " "
        final = []
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
        return final



    def encode(self, seq, seq_length):
        return self.apply_meanpool(seq, seq_length)
        filters1 = tf.get_variable('conv1', [1, self.config.word_embed_size, self.config.word_embed_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        #conv1_b = tf.get_variable('conv1_b', initializer=tf.contrib.layers.xavier_initializer(), shape=self.config.hidden_size)
        conv_layer1 = tf.nn.conv1d(seq, filters1, 1, padding='SAME')
        #conv_layer1 = tf.nn.relu(tf.nn.conv1d(seq, filters1, 1, padding='SAME')+conv1_b)
        '''

        filters2 = tf.get_variable('conv2', [1, self.config.hidden_size, self.config.hidden_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        conv2_b = tf.get_variable('conv2_b', initializer=tf.contrib.layers.xavier_initializer(), shape=self.config.hidden_size)
        self.conv_layer2 = tf.nn.relu(tf.nn.conv1d(conv_layer1, filters2, 1, padding='SAME')+conv2_b)
        '''

#        cell = tf.contrib.rnn.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
#        _, state = tf.nn.dynamic_rnn(cell, seq, dtype=tf.float32, sequence_length=seq_length)

        with tf.variable_scope('fw'):
            cell_fw = tf.contrib.rnn.GRUCell(self.config.hidden_size/2, activation=tf.nn.tanh)
        with tf.variable_scope('bw'):
            cell_bw = tf.contrib.rnn.GRUCell(self.config.hidden_size/2, activation=tf.nn.tanh)
        _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, conv_layer1, dtype=tf.float32, sequence_length=seq_length)
        #_, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, seq, dtype=tf.float32, sequence_length=seq_length)
        state = tf.concat(states, 1)
        #_, state = tf.nn.dynamic_rnn(cell, self.conv_layer2, dtype=tf.float32, sequence_length=seq_length)

        layer1 = tf.nn.relu(linear('sm_layer1', state, [self.config.hidden_size, self.config.layer1_size]))
        layer2 = tf.nn.relu(linear('sm_layer2', layer1, [self.config.layer1_size, self.config.layer2_size]))
        #layer3 = linear('sm_layer3', layer2, [self.config.layer2_size, 128], self.phase)
        layer3 = tf.nn.relu(linear('sm_layer3', layer2, [self.config.layer2_size, self.config.layer3_size]))
        layer4 = tf.nn.relu(linear('sm_layer4', layer3, [self.config.layer3_size, self.config.layer4_size]))
        #layer3 = tf.nn.l2_normalize(tf.nn.relu(linear('sm_layer3', layer2, [self.config.layer2_size, self.config.layer3_size], self.phase)), dim=1)

#        return layer4 #tf.nn.l2_normalize(layer4, dim=1)
        return tf.nn.l2_normalize(layer4, dim=1)

    def get_score(self, embedding):
        ################ Experiment with the design here:
        '''
        self.last_layer1 =  tf.nn.tanh(linear('last_layer1', self.z, [128, self.config.layer2_size]))
        self.last_layer2 =  tf.nn.tanh(linear('last_layer2', self.last_layer1, [self.config.layer2_size, self.config.layer2_size]))
        self.last_layer3 =  (linear('last_layer3', self.last_layer2, [self.config.layer2_size, self.config.layer2_size]))
        last_layer2 = self.last_layer3
        '''
        '''
        self.last_layer1 =  tf.nn.relu(linear_sparse('last_layer1', self.ancestry_sparse_tensor, [self.config.hpo_size, self.config.layer2_size]))
        last_layer2 =  linear('last_layer2', self.last_layer1, [self.config.layer2_size, self.config.layer2_size], self.phase)
        '''
 #       '''
        self.w = tf.get_variable("last_layerWW", shape = [self.config.concepts_size, self.config.layer4_size], initializer = tf.random_normal_initializer(stddev=0.1))
        last_layer_b = tf.get_variable('last_layer'+"B", shape = [self.config.concepts_size], initializer = tf.random_normal_initializer(stddev=0.001))
        self.aggregated_w = tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, self.w) 
        #self.last_layer_w =  linear_sparse('last_layerW', self.ancestry_sparse_tensor, [self.config.hpo_size, self.config.layer4_size])
#        last_layer_w =  linear('last_l', tf.nn.relu(last_layer_w_p), [self.config.layer3_size, self.config.layer4_size])
        #last_layer_w = tf.get_variable('last_layer'+"W", shape = [self.config.hpo_size, self.config.layer4_size], initializer = tf.random_normal_initializer(stddev=0.1))

        score_layer = tf.matmul(embedding, tf.transpose(self.aggregated_w))  + last_layer_b
 #       '''
        #score_layer = tf.reduce_sum(embedding*embedding, axis=1, keep_dims=True) + tf.transpose(tf.reduce_sum(self.z*self.z, axis=1, keep_dims=True)) -2*tf.matmul(embedding, self.z, transpose_b=True)
#        score_layer = linear('last_layer', embedding, [self.config.layer3_size, self.config.hpo_size]) 
 #       w = tf.get_variable('last_layer'+"W", shape = [self.config.layer3_size, self.config.hpo_size], initializer = tf.random_normal_initializer(stddev=0.1))
 #       b = tf.get_variable('last_layer'+"B", shape = [self.config.hpo_size], initializer = tf.random_normal_initializer(stddev=0.01))
 #       score_layer = tf.matmul(embedding, w) + b
        return score_layer

    #############################
    ##### Creates the model #####
    #############################
    def __init__(self, config, ont, word_model):
        tf.reset_default_graph()
        self.ont = ont
        self.word_model = word_model
        config.update_with_reader(self.ont)
        self.config = config
        '''
        if ancs_sparse is None:
            self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
        else:
            self.ancestry_sparse_tensor = tf.sparse_reorder(tf.SparseTensor(indices = ancs_sparse, values = [1.0]*len(ancs_sparse), dense_shape=[config.hpo_size, config.hpo_size]))
        '''

        ### Inputs ###
        self.label = tf.placeholder(tf.int32, shape=[None])
        self.seq = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, config.word_embed_size])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])
        self.keep_prob = tf.placeholder(tf.float32)

        self.ancestry_sparse_tensor = tf.sparse_reorder(tf.SparseTensor(indices = ont.sparse_ancestrs, values = [1.0]*len(ont.sparse_ancestrs), dense_shape=[config.concepts_size, config.concepts_size]))

#        self.anchors = tf.get_variable('anchors', [19202, config.layer4_size], trainable=False) #19k x 1024 , B x 1024

        label_one_hot = tf.one_hot(self.label, config.concepts_size)

        self.seq_embedding = self.encode(self.seq, self.seq_len)

        with tf.variable_scope('top_cnn') as scope:
            self.score_layer = self.get_score(self.seq_embedding)
        self.pred = tf.nn.softmax(self.score_layer)
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label_one_hot, self.score_layer)) 

	self.lr = tf.Variable(config.lr, trainable=False)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.wins = [tf.nn.max_pool(tf.expand_dims(self.layer2, -1), [1, k, 1, 1], [1, 1, 1, 1], 'VALID') for k in range(1,8)]
        with tf.variable_scope('top_cnn') as scope:
            scope.reuse_variables()
            #self.wins = [tf.nn.relu(linear('lassst', x, [self.config.cl2, self.config.cl2])) for x in self.wins]
            self.wins = [tf.nn.l2_normalize(tf.nn.relu(linear('lassst', tf.reshape(x, [-1, self.config.cl2]), [self.config.cl2, self.config.cl2])), dim=1) for x in self.wins]
            self.wins = [tf.nn.softmax(self.get_score(x)) for x in self.wins]
            self.wins = [tf.reshape(x,[-1,config.max_sequence_length-i, self.config.concepts_size]) for i, x in enumerate(self.wins)]



        print "starting session"
	self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        print "initializing"
	self.sess.run(tf.global_variables_initializer())
        print "initialized"
        self.anchors_set = False


    #########################################################

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
            none_id = len(self.ont.concepts)
            raw_samples+=neg_samples
            labels += [none_id]*len(neg_samples)

        self.training_samples = {}
        self.training_samples['seq'], self.training_samples['seq_len'] = self.phrase2vec(raw_samples, self.config.max_sequence_length)
        self.training_samples['label'] = np.array(labels)

    def train_epoch(self):
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
                    self.keep_prob:self.config.keep_prob,\
                    self.label:batch['label']} 
            _ , batch_loss = self.sess.run([self.train_step, self.loss], feed_dict = batch_feed)

            report_loss += batch_loss
            total_loss += batch_loss
            if ct % report_len == report_len-1:
                print "Step =", ct+1, "\tLoss =", report_loss/report_len
                #sys.stdout.flush()
                report_loss = 0
            ct += 1

        return total_loss/ct

    '''
    def set_anchors(self):
        syns = []
        syns_labels = []
        for i,hpid in enumerate(self.rd.concepts):
            for s in self.rd.names[hpid]:
                syns.append(s)
                syns_labels.append(i)

        self.anchors_set = True
	header = 0
	batch_size = 512

        all_vecs = []
	while header < len(syns):
            raw_batch = syns[header:min(header+batch_size, len(syns))]
            batch = self.rd.create_test_sample(raw_batch)
            header += batch_size
            querry_dict = {self.seq : batch['seq'], self.seq_len: batch['seq_len']}
            all_vecs.append(self.sess.run(self.seq_embedding, feed_dict = querry_dict))
            print len(all_vecs), all_vecs[-1].shape

        self.anchors = tf.get_variable('anchors', [19203, 1024], trainable=False) #19k x 1024 , B x 1024
        #self.anchors = tf.get_variable('anchors', [19202, config.layer4_size], trainable=False) #19k x 1024 , B x 1024
        self.sess.run(tf.assign(self.anchors, np.concatenate(all_vecs)))
        self.anchors_label = syns_labels
        self.anchors_dis = tf.matmul(self.seq_embedding, self.anchors, transpose_b=True)
        
    def get_hp_id_from_anchor(self, querry, count=1):
        inp = self.rd.create_test_sample(querry)

        querry_dict = {self.seq : inp['seq'], self.seq_len: inp['seq_len']}
        res_querry = self.sess.run(self.anchors_dis, feed_dict = querry_dict)
        results=[]
        for s in range(len(querry)):
            indecies_querry = np.argsort(-res_querry[s,:])

            tmp_res = []
            for i in indecies_querry:
       #       comment start  
                print i
                if i == len(self.rd.concepts):
                    tmp_res.append(('None',res_querry[i]))
                else:
       #        comment end
                hp_index = self.anchors_label[i]
                tmp_res.append((self.rd.concepts[hp_index],res_querry[s,i]))
                if len(tmp_res)>=count:
                        break
            results.append(tmp_res)
        return results
    '''

    def get_hp_id(self, querry, count=1):
        if self.anchors_set:
            return self.get_hp_id_from_anchor(querry, count)
        seq, seq_len = self.phrase2vec(querry, self.config.max_sequence_length)

        querry_dict = {self.seq : seq, self.seq_len: seq_len, self.keep_prob:1.0}
        res_querry = self.sess.run(self.pred, feed_dict = querry_dict)

        results=[]
        for s in range(len(querry)):
            indecies_querry = np.argsort(-res_querry[s,:])

            tmp_res = []
            for i in indecies_querry:
                '''
                print i
                if i == len(self.rd.concepts):
                    tmp_res.append(('None',res_querry[i]))
                else:
                '''
                tmp_res.append((self.ont.concepts[i],res_querry[s,i]))
                if len(tmp_res)>=count:
                        break
            results.append(tmp_res)

        return results

