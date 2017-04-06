import tensorflow as tf
import numpy as np


def weight_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def linear(name, x, shape, phase=0):
#    w = tf.get_variable(name+"W", shape, initializer=tf.contrib.layers.xavier_initializer())
    w = tf.get_variable(name+"W", shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))
#    w = weight_variable(name+"W", shape)
    b = tf.get_variable(name+"B", shape = shape[1], initializer = tf.random_normal_initializer(stddev=0.1))
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

    def apply_meanpool(self, seq, seq_length):
        ################ Experiment with the design here:
        filters1 = tf.get_variable('conv1', [1, self.config.word_embed_size, self.config.hidden_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        conv1_b = tf.get_variable('conv1_b', initializer=tf.contrib.layers.xavier_initializer(), shape=self.config.hidden_size)
        conv_layer1 = tf.nn.relu(tf.nn.conv1d(seq, filters1, 1, padding='SAME')+conv1_b)

        filters2 = tf.get_variable('conv2', [1, self.config.hidden_size, self.config.hidden_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        conv2_b = tf.get_variable('conv2_b', initializer=tf.contrib.layers.xavier_initializer(), shape=self.config.hidden_size)
        self.conv_layer2 = tf.nn.relu(tf.nn.conv1d(conv_layer1, filters2, 1, padding='SAME')+conv2_b)

        cell = tf.contrib.rnn.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)
        _, state = tf.nn.dynamic_rnn(cell, self.conv_layer2, dtype=tf.float32, sequence_length=seq_length)
        return state

        return tf.reduce_max(self.conv_layer2, [1])


    def encode(self, seq, seq_length):
        ################ Experiment with the design here:
        embed1 = self.apply_meanpool(seq, seq_length)
        layer1 = tf.nn.relu(linear('sm_layer1', embed1, [self.config.hidden_size, self.config.layer1_size], self.phase))
        layer2 = tf.nn.relu(linear('sm_layer2', layer1, [self.config.layer1_size, self.config.layer2_size], self.phase))
        #layer3 = linear('sm_layer3', layer2, [self.config.layer2_size, 128], self.phase)
        layer3 = tf.nn.relu(linear('sm_layer3', layer2, [self.config.layer2_size, self.config.layer3_size], self.phase))
        #layer3 = tf.nn.l2_normalize(tf.nn.relu(linear('sm_layer3', layer2, [self.config.layer2_size, self.config.layer3_size], self.phase)), dim=1)
        return layer3

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
        '''
        self.last_layer1 =  linear_sparse('last_layer1', self.ancestry_sparse_tensor, [self.config.hpo_size, self.config.layer2_size])
        last_layer2 = self.last_layer1
        score_layer = tf.matmul(embedding, tf.transpose(last_layer2))#  + last_layer_b
        '''
        #score_layer = tf.reduce_sum(embedding*embedding, axis=1, keep_dims=True) + tf.transpose(tf.reduce_sum(self.z*self.z, axis=1, keep_dims=True)) -2*tf.matmul(embedding, self.z, transpose_b=True)
        score_layer = linear('last_layer', embedding, [self.config.layer3_size, self.config.hpo_size]) 
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
#        self.z = tf.get_variable("ancestry_z", [config.hpo_size, 128], trainable=False)
        self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
        self.input_sequence = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, config.word_embed_size])
        self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
        self.phase = tf.placeholder(tf.bool) 

        label = tf.one_hot(self.input_hpo_id, config.hpo_size)

        input_embedding = self.encode(self.input_sequence, self.input_sequence_lengths)

        self.score_layer = self.get_score(input_embedding)
        self.pred = tf.nn.softmax(self.score_layer)
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label, self.score_layer)) 
        return
        self.score_layer = self.get_score(input_embedding)
        self.pred = tf.nn.softmax(self.score_layer)
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label, self.score_layer)) 

