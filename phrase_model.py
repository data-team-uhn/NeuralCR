import tensorflow as tf
import numpy as np


def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

'''
def weight_variable(name, shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
'''

def linear(name, x, shape):
    w = weight_variable(name+"W", shape)
    b = weight_variable(name+"B", [shape[1]])
    return tf.matmul(x,w) + b


def embedding_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.random_normal_initializer(stddev=0.1))

class NCRModel():
    def get_HPO_embedding(self, indices=None):
        embedding = self.HPO_embedding
        if indices is not None:
            embedding = tf.gather(self.HPO_embedding, indices)
            return embedding #tf.maximum(0.0, embedding)

    def apply_meanpool(self, seq, seq_length):
        inputs = tf.reshape(seq, [-1, self.config.word_embed_size])
            
        '''
        w1 = weight_variable('layer1W', [self.config.word_embed_size, self.config.l1_size])
        b1 = weight_variable('layer1B', [self.config.l1_size])
        w2 = weight_variable('layer2W', [self.config.l1_size, self.config.l2_size])
        b2 = weight_variable('layer2B', [self.config.l2_size])
        w3 = weight_variable('layer3W', [l3_size, l3_size])
        b3 = weight_variable('layer3B', [l3_size])
        '''

        layer1 = linear('mp_l1', inputs, [self.config.word_embed_size, self.config.l1_size])
        layer2 = linear('mp_l2', layer1, [self.config.l1_size, self.config.l2_size])
        layer3 = linear('mp_l3', layer2, [self.config.l2_size, self.config.hidden_size])

        layer3_reshaped = tf.reshape(layer3, [-1, self.config.max_sequence_length, self.config.hidden_size])
        return tf.reduce_max(layer3_reshaped, [1])


    def apply_rnn(self, seq, seq_length):
        #		seq_embeded = tf.nn.embedding_lookup(self.word_embedding, seq)
        #		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.config.max_sequence_length, seq_embeded)]
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.config.max_sequence_length, seq)]
        #'''
        w1 = weight_variable('layer1W', [self.config.word_embed_size, self.config.l1_size])
        b1 = weight_variable('layer1B', [self.config.l1_size])
        w2 = weight_variable('layer2W', [self.config.l1_size, self.config.l2_size])
        b2 = weight_variable('layer2B', [self.config.l2_size])
        '''
        w3 = weight_variable('layer3W', [self.config.l2_size, self.config.l3_size])
        b3 = weight_variable('layer3B', [self.config.l3_size])
        '''

        mlp_inputs = [tf.nn.tanh(tf.matmul(x, w1)+b1) for x in inputs]
        mlp_inputs = [tf.nn.tanh(tf.matmul(x, w2)+b2) for x in mlp_inputs]
#        mlp_inputs = [tf.nn.tanh(tf.matmul(x, w3)+b3) for x in mlp_inputs]
        #'''
        cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size, activation=tf.nn.tanh)

        #		_, state = tf.nn.rnn(cell, inputs, dtype=tf.float32, sequence_length=seq_length)
        _, state = tf.nn.rnn(cell, mlp_inputs, dtype=tf.float32, sequence_length=seq_length)
        return state

        layer1 = tf.nn.softplus(linear('layer1', state, [self.config.hidden_size, self.config.hidden_size]))
        layer2 = tf.nn.softplus(linear('layer2', state, [self.config.hidden_size, self.config.hidden_size]))
        return tf.nn.tanh(linear('layer3', state, [self.config.hidden_size, self.config.concept_size]))

        return #		return tf.nn.rnn(cell, inputs, tf.gather(self.type_embedding, self.input_type_id), dtype=tf.float32, sequence_length=seq_length)

	#############################
	##### Creates the model #####
	#############################
    def __init__(self, config, training = False):
        self.config = config
        self.ancestry_masks = tf.get_variable("ancestry_masks", [config.hpo_size, config.hpo_size], trainable=False)

        ### Inputs ###
        self.input_hpo_id = tf.placeholder(tf.int32, shape=[None])
        self.input_sequence = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, config.word_embed_size])
        self.input_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
        label = tf.one_hot(self.input_hpo_id, config.hpo_size)

        gru_state = self.apply_rnn(self.input_sequence, self.input_sequence_lengths) 

        layer1 = tf.nn.tanh(linear('sm_layer1', gru_state, [self.config.hidden_size, self.config.layer1_size]))
        layer2 = tf.nn.tanh(linear('sm_layer2', layer1, [self.config.layer1_size, self.config.layer2_size]))
        layer3 = tf.nn.tanh(linear('sm_layer3', layer2, [self.config.layer2_size, self.config.layer3_size]))
        layer4= (linear('sm_layer4', layer3, [self.config.layer3_size, self.config.hpo_size]))

        mixing_w = tf.Variable(1.0)

        score_layer = (mixing_w * layer4 +\
                tf.matmul(layer4, tf.transpose(self.ancestry_masks)))

        self.pred = tf.nn.softmax(score_layer)

        if training:
            self.loss = tf.reduce_mean(\
                    tf.nn.softmax_cross_entropy_with_logits(score_layer, label))


