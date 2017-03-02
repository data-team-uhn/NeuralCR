import tensorflow as tf
import numpy as np


def weight_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def linear(name, x, shape):
#    w = tf.get_variable(name+"W", shape, initializer=tf.contrib.layers.xavier_initializer())
    w = tf.get_variable(name+"W", shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))
#    w = weight_variable(name+"W", shape)
    b = tf.get_variable(name+"B", shape = shape[1], initializer = tf.random_normal_initializer(stddev=0.1))
#    b = weight_variable(name+"B", [shape[1]])
    return tf.matmul(x,w) + b

def linear_sparse(name, x, shape):
#    w = tf.get_variable(name+"W", shape, initializer=tf.contrib.layers.xavier_initializer())
    w = tf.get_variable(name+"W", shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))
#    w = weight_variable(name+"W", shape)
    b = tf.get_variable(name+"B", shape = shape[1], initializer = tf.random_normal_initializer(stddev=0.1))
#    b = weight_variable(name+"B", [shape[1]])
    return tf.sparse_tensor_dense_matmul(x,w) + b


def embedding_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))

class NCRModel():
    def get_HPO_embedding(self, indices=None):
        embedding = self.HPO_embedding
        if indices is not None:
            embedding = tf.gather(self.HPO_embedding, indices)
            return embedding #tf.maximum(0.0, embedding)

    def apply_rnn(self, seq, seq_length):
            #seq_embeded = tf.nn.embedding_lookup(self.word_embedding, seq)
#		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.config.max_sequence_length, seq_embeded)]
            inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(seq, self.config.max_sequence_length, 1)]
            #inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.config.max_sequence_length, seq)]
            w1 = weight_variable('layer1W', [self.config.word_embed_size, self.config.hidden_size])
            b1 = weight_variable('layer1B', [self.config.hidden_size])
            w2 = weight_variable('layer2W', [self.config.hidden_size, self.config.hidden_size])
            b2 = weight_variable('layer2B', [self.config.hidden_size])
            '''
            w3 = weight_variable('layer3W', [l3_size, l3_size])
            b3 = weight_variable('layer3B', [l3_size])
            '''
            '''

            mlp_inputs = [tf.nn.tanh(tf.matmul(x, w1)+b1) for x in inputs]
            mlp_inputs = [tf.nn.tanh(tf.matmul(x, w2)+b2) for x in mlp_inputs]
            #mlp_inputs = [tf.nn.tanh(tf.matmul(x, w3)+b3) for x in mlp_inputs]
            '''
            mlp_inputs = [tf.nn.relu(tf.matmul(x, w1)+b1) for x in inputs]
            mlp_inputs = [tf.nn.relu(tf.matmul(x, w2)+b2) for x in mlp_inputs]
            cell = tf.contrib.rnn.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
            
            
            init_state = embedding_variable('rnn_init_state', [1,self.config.hidden_size])
            _, state = tf.nn.dynamic_rnn(cell, seq, dtype=tf.float32, sequence_length=seq_length)
            #_, state = tf.nn.dynamic_rnn(cell, mlp_inputs, tf.tile(init_state, [tf.shape(mlp_inputs[0])[0],1]), dtype=tf.float32, sequence_length=seq_length)
#            _, state = tf.nn.rnn(cell, inputs, dtype=tf.float32, sequence_length=seq_length)
            #_, state = tf.nn.rnn(cell, mlp_inputs, dtype=tf.float32, sequence_length=seq_length)
            return state

    def apply_meanpool(self, seq, seq_length):
        #filters1 = tf.get_variable('conv1', [1, self.config.word_embed_size, self.config.hidden_size], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
        filters1 = tf.get_variable('conv1', [1, self.config.word_embed_size, self.config.hidden_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        #conv1_b = tf.get_variable('conv1_b', initializer=tf.random_normal_initializer(stddev=0.1), shape=self.config.hidden_size)
        conv1_b = tf.get_variable('conv1_b', initializer=tf.contrib.layers.xavier_initializer(), shape=self.config.hidden_size)
        conv_layer1 = tf.nn.relu(tf.nn.conv1d(seq, filters1, 1, padding='SAME')+conv1_b)

        #filters2 = tf.get_variable('conv2', [1, self.config.hidden_size, self.config.hidden_size], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
        filters2 = tf.get_variable('conv2', [1, self.config.hidden_size, self.config.hidden_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        #conv2_b = tf.get_variable('conv2_b', initializer=tf.random_normal_initializer(stddev=0.1), shape=self.config.hidden_size)
        conv2_b = tf.get_variable('conv2_b', initializer=tf.contrib.layers.xavier_initializer(), shape=self.config.hidden_size)
        self.conv_layer2 = tf.nn.relu(tf.nn.conv1d(conv_layer1, filters2, 1, padding='SAME')+conv2_b)

        self.mask = tf.tile(tf.expand_dims(tf.sequence_mask(self.input_sequence_lengths, self.config.max_sequence_length), axis=2) , tf.pack([1,1,self.config.hidden_size]))
        self.conv_layer2 = tf.select(self.mask, self.conv_layer2, tf.zeros_like(self.conv_layer2))


        return tf.reduce_max(self.conv_layer2, [1])
#        return tf.nn.l2_normalize(tf.reduce_sum(self.conv_layer2, [1]), dim=1)
        #return tf.reduce_sum(layer2, [1])


    def encode(self, seq, seq_length):
        embed1 = self.apply_rnn(seq, seq_length)
        #embed2 = tf.nn.tanh(linear('encode2', embed1, [self.config.hidden_size, self.config.layer1_size]))
        #embed3 = linear('encode3', embed2, [self.config.layer1_size, self.config.layer2_size])
        #return tf.nn.l2_normalize(embed3, dim=1)

        #embedding = = self.apply_meanpool(seq, seq_length) 

        layer1 = tf.nn.relu(linear('sm_layer1', embed1, [self.config.hidden_size, self.config.layer1_size]))
#        self.layer2 = tf.nn.relu(linear('sm_layer2', layer1, [self.config.layer1_size, self.config.layer2_size]))
        layer2 = tf.nn.relu(linear('sm_layer2', layer1, [self.config.layer1_size, self.config.layer2_size]))
#        layer3 = tf.nn.tanh(linear('sm_layer3', layer2, [self.config.layer2_size, self.config.layer3_size]))
        layer3 = tf.nn.l2_normalize(tf.nn.relu(linear('sm_layer3', layer1, [self.config.layer2_size, self.config.layer3_size])), dim=1)
#        layer3 = tf.nn.relu(linear('sm_layer3', layer1, [self.config.layer2_size, self.config.layer3_size]))
        return layer3

    def get_score(self, embedding):
        '''
        init_w1 = tf.constant((np.random.normal(0.0,0.1,[self.config.hpo_size, self.config.layer2_size])), dtype=tf.float32)
        last_layer_w1 = tf.get_variable('last_layer_w', initializer=init_w1)
        init_b1 = tf.constant((np.random.normal(0.0,0.1,[self.config.layer2_size])), dtype=tf.float32)
        last_layer_b1 = tf.get_variable('last_layer_b', initializer=init_b1)

        init_w1b = tf.constant((np.random.normal(0.0,0.1,[self.config.hpo_size, 512])), dtype=tf.float32)
        last_layer_w1b = tf.get_variable('last_layer_wb', initializer=init_w1)
        '''

        #self.last_layer1a = (tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, last_layer_w1) + last_layer_b1)
        #last_layer2 = self.last_layer1a
        #last_layer2 = tf.nn.l2_normalize(self.last_layer1a, dim=1)
#        self.last_layer1b =  tf.transpose(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.ancestry_sparse_tensor), tf.transpose(last_layer_w1b)))
#        self.last_layer1 =  tf.nn.relu(self.last_layer1a + self.last_layer1b + last_layer_b1)
#        self.last_layer1 =  (linear_sparse('last_layer1', self.ancestry_sparse_tensor, [self.config.hpo_size, self.config.layer2_size]))
        self.last_layer1 =  tf.nn.relu(linear_sparse('last_layer1', self.ancestry_sparse_tensor, [self.config.hpo_size, self.config.layer2_size]))
        #self.last_layer1 =  tf.nn.tanh(linear_sparse('last_layer1', self.ancestry_sparse_tensor, [self.config.hpo_size, self.config.layer2_size]))
        #last_layer2 = self.last_layer1
        #self.last_layer1 =  tf.nn.relu(tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, last_layer_w1)+last_layer_b1)
        last_layer2 =  linear('last_layer2', self.last_layer1, [self.config.layer2_size, self.config.layer2_size] )
#        last_layer2 =  tf.nn.l2_normalize(linear('last_layer2', self.last_layer1, [self.config.layer2_size, self.config.layer2_size] ), dim=1)

        #last_layer2 =  tf.transpose(tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, tf.transpose(last_layer_w1)))
#        proc_last_layer_w =  tf.nn.tanh(tf.transpose(tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, tf.transpose(last_layer_w))))#, dim=0)
        #proc_last_layer_w =  tf.nn.l2_normalize(tf.transpose(tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, tf.transpose(last_layer_w))), dim=0)

#        self.layer4= (linear('sm_layer4', self.layer2, [self.config.layer2_size, self.config.hpo_size]))
#        score_layer = linear('vanila', embedding, [self.config.layer2_size, self.config.hpo_size])#  + last_layer_b
        score_layer = tf.matmul(embedding, tf.transpose(last_layer2))#  + last_layer_b
        return score_layer



    #############################
    ##### Creates the model #####
    #############################
    def __init__(self, config, training = False, ancs_sparse = None):
        self.config = config

        if ancs_sparse is None:
            self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)
        else:
            self.ancestry_sparse_tensor = tf.sparse_reorder(tf.SparseTensor(indices = ancs_sparse, values = [1.0]*len(ancs_sparse), dense_shape=[config.hpo_size, config.hpo_size]))

        ### Inputs ###
        self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
        self.input_sequence = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, config.word_embed_size])
        self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
        label = tf.one_hot(self.input_hpo_id, config.hpo_size)

        input_embedding = self.encode(self.input_sequence, self.input_sequence_lengths)
        self.score_layer = self.get_score(input_embedding)

       

        '''
        self.score_layer = (mixing_w * self.layer4 + tf.minimum(self.layer4, tf.zeros_like(self.layer4)) +\
                tf.matmul(tf.maximum(self.layer4, tf.zeros_like(self.layer4)), tf.transpose(self.ancestry_masks)))
        '''
        ### TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  ancestry_sparse_matrix
        '''
        if ancs_sparse is None:
            self.score_layer = (mixing_w * self.layer4 + tf.minimum(self.layer4, tf.zeros_like(self.layer4)) +\
                    tf.matmul(tf.maximum(self.layer4, tf.zeros_like(self.layer4)), tf.transpose(self.ancestry_masks)))
#            self.score_layer =  tf.matmul(self.layer4, tf.transpose(self.ancestry_masks))
        else:
#            self.score_layer =  self.layer4
#            self.score_layer = mixing_w * self.layer4 - tf.nn.relu(-self.layer4) +\
#                    tf.transpose(tf.sparse_tensor_dense_matmul(ancestry_sparse_tensor, tf.transpose(tf.nn.relu(self.layer4))))
            self.score_layer = tf.transpose(tf.sparse_tensor_dense_matmul(ancestry_sparse_tensor, tf.transpose(self.layer4)))
#            self.score_layer = mixing_w*self.layer4 + tf.transpose(tf.sparse_tensor_dense_matmul(ancestry_sparse_tensor, tf.transpose(self.layer4)))

        '''
        self.pred = tf.nn.softmax(self.score_layer)
        #self.pred = self.score_layer
        '''
        self.pred1 = tf.nn.softmax(self.score_layer)
        self.pred2 = tf.nn.softmax(self.layer4)
        tf.concat(self.pred1, self.pred2)
        '''
        #self.pred = tf.nn.softmax(self.score_layer + last_layer_b)

        if training:
            l2_w = 0.0
           # self.loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(self.config.alpha - tf.expand_dims(tf.reduce_sum(self.score_layer*label, [1]),1) + self.score_layer, 0.0), [-1]))
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label, self.score_layer)) # + l2_w * tf.reduce_sum(tf.nn.relu(last_layer_w_para))
                    #tf.nn.softmax_cross_entropy_with_logits(self.score_layer, label)) # + l2_w * tf.reduce_sum(tf.nn.relu(last_layer_w_para))

