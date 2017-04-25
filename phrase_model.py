import tensorflow as tf
import numpy as np


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

class NCRModel():
    def get_HPO_embedding(self, indices=None):
        embedding = self.HPO_embedding
        if indices is not None:
            embedding = tf.gather(self.HPO_embedding, indices)
            return embedding #tf.maximum(0.0, embedding)

    def encode(self, seq, seq_length):
        '''
        filters1 = tf.get_variable('conv1', [1, self.config.word_embed_size, self.config.word_embed_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        #conv1_b = tf.get_variable('conv1_b', initializer=tf.contrib.layers.xavier_initializer(), shape=self.config.hidden_size)
        conv_layer1 = tf.nn.conv1d(seq, filters1, 1, padding='SAME')
        #conv_layer1 = tf.nn.relu(tf.nn.conv1d(seq, filters1, 1, padding='SAME')+conv1_b)

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
        #_, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, conv_layer1, dtype=tf.float32, sequence_length=seq_length)
        _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, seq, dtype=tf.float32, sequence_length=seq_length)
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
        last_layer_w =  linear_sparse('last_layerW', self.ancestry_sparse_tensor, [self.config.hpo_size, self.config.layer4_size])
#        last_layer_w =  linear('last_l', tf.nn.relu(last_layer_w_p), [self.config.layer3_size, self.config.layer4_size])
        #last_layer_w = tf.get_variable('last_layer'+"W", shape = [self.config.hpo_size, self.config.layer4_size], initializer = tf.random_normal_initializer(stddev=0.1))
        last_layer_b = tf.get_variable('last_layer'+"B", shape = [self.config.hpo_size], initializer = tf.random_normal_initializer(stddev=0.001))

        score_layer = tf.matmul(embedding, tf.transpose(last_layer_w))  + last_layer_b
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
    def __init__(self, config, rd):
        tf.reset_default_graph()
        self.rd = rd
        config.update_with_reader(self.rd)
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

        self.ancestry_sparse_tensor = tf.sparse_reorder(tf.SparseTensor(indices = rd.sparse_ancestrs, values = [1.0]*len(rd.sparse_ancestrs), dense_shape=[config.hpo_size, config.hpo_size]))

#        self.anchors = tf.get_variable('anchors', [19202, config.layer4_size], trainable=False) #19k x 1024 , B x 1024

        label_one_hot = tf.one_hot(self.label, config.hpo_size)

        self.seq_embedding = self.encode(self.seq, self.seq_len)

        self.score_layer = self.get_score(self.seq_embedding)
        self.pred = tf.nn.softmax(self.score_layer)
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label_one_hot, self.score_layer)) 

	self.lr = tf.Variable(config.lr, trainable=False)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

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

    def train_epoch(self):
	self.rd.reset_counter()
	ct = 0
	report_loss = 0
	total_loss = 0
        report_len = 20
	while True:
            batch = self.rd.read_batch(self.config.batch_size)
            if batch == None:
                break

            batch_feed = {self.seq:batch['seq'],\
                    self.seq_len:batch['seq_len'],\
                    self.label:batch['hp_id']} 
            _ , batch_loss = self.sess.run([self.train_step, self.loss], feed_dict = batch_feed)

            report_loss += batch_loss
            total_loss += batch_loss
            if ct % report_len == report_len-1:
                print "Step =", ct+1, "\tLoss =", report_loss/report_len
                #sys.stdout.flush()
                report_loss = 0
            ct += 1

        return total_loss/ct

    def set_anchors(self, syns, syns_labels):
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
                '''
                print i
                if i == len(self.rd.concepts):
                    tmp_res.append(('None',res_querry[i]))
                else:
                '''
                hp_index = self.anchors_label[i]
                tmp_res.append((self.rd.concepts[hp_index],res_querry[s,i]))
                if len(tmp_res)>=count:
                        break
            results.append(tmp_res)
        return results

    def get_hp_id(self, querry, count=1):
        if self.anchors_set:
            return self.get_hp_id_from_anchor(querry, count)
        inp = self.rd.create_test_sample(querry)

        querry_dict = {self.seq : inp['seq'], self.seq_len: inp['seq_len']}
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
                tmp_res.append((self.rd.concepts[i],res_querry[s,i]))
                if len(tmp_res)>=count:
                        break
            results.append(tmp_res)

        return results

