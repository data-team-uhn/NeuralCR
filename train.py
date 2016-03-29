import tensorflow as tf
import cPickle as pickle
import numpy as np
from triplet_reader import DataReader
# from concept2vec import concept_vector_model
import ncr_cnn_model
import sys


def read_files():
    directory="./train_data_gen/data_files/"
    files = ['validation_synonym_3_5_triplets', 'validation_synonym_3_5_labels', 'validation_synonym_2_2_triplets',
             'validation_synonym_2_2_labels', 'validation_synonym_1_1_triplets', 'validation_synonym_1_1_labels',
             'validation_graph_3_5_triplets', 'validation_graph_3_5_labels', 'test_graph_2_2_triplets',
             'test_graph_2_2_labels', 'test_synonym_3_5_triplets', 'test_synonym_3_5_labels',
             'test_synonym_2_2_triplets', 'test_synonym_2_2_labels', 'test_synonym_1_1_triplets',
             'test_synonym_1_1_labels', 'training_triplets', 'training_labels',
             'test_graph_3_5_triplets', 'test_graph_3_5_labels', 'validation_graph_2_2_triplets',
             'validation_graph_2_2_labels']
    data ={}
    for f in files:
        data[f] = np.load(directory+'/'+f+".npy")
    return data
    # validation_synonym_triplets = np.load(directory+'/validation_synonym_triplets.npy')


class firstTrainConfig():
    lr_decay=0.8
    lr_init=1e-5
    batch_size = 100


class TrainingUnit():
    def __init__(self, modelConfig, trainConfig):
        self.trainConfig = trainConfig
        with tf.variable_scope("models"):
            self.guide = ncr_cnn_model.NCRModel(modelConfig)
        with tf.variable_scope("models", reuse=True):
            self.concept0 = ncr_cnn_model.NCRModel(modelConfig)
            self.concept1 = ncr_cnn_model.NCRModel(modelConfig)

        y_ = tf.concat(1, [tf.reduce_sum (self.guide.rep * self.concept0.rep, 1, keep_dims=True), tf.reduce_sum (self.guide.rep * self.concept1.rep, 1, keep_dims=True)])
        self.y_normalised = y_ / tf.reduce_sum(y_, 1, True)
        self.y = tf.placeholder(tf.float32, [None, 2])

        self.loss = - tf.reduce_sum(self.y * tf.log(self.y_normalised))

        correct_prediction = tf.equal(tf.arg_max(self.y, 1), tf.arg_max(self.y_normalised,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.lr = tf.Variable( 0.0, trainable=False)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def feed_input(self, var, sess, input_set):
        return sess.run(var, feed_dict={self.guide.input_vectors:input_set[0][:,:,0], self.concept0.input_vectors:input_set[0][:,:,1], self.concept1.input_vectors:input_set[0][:,:,2], self.y:input_set[1]})

    def run_epoch(self, sess, reader, epoch, validation_sets, all_training):
        lr_new = self.trainConfig.lr_init * (self.trainConfig.lr_decay ** max(epoch-4.0, 0.0))
        sess.run(tf.assign(self.lr, lr_new))

        print "Epoch ::", epoch, "\tLearning Rate ::", lr_new

        for val_set in validation_sets:
            print val_set + " Accuracy :: ", self.feed_input(self.accuracy, sess, validation_sets[val_set])
		# print "Training Accuracy :: ", self.feed_input(self.accuracy, sess, all_training)

        step=0
        while True:
            new_batch, labels = reader.read_batch('training_triplets', 'training_labels', self.trainConfig.batch_size)
            if new_batch == labels:
                break
            if step%100 == 0:
                print "Step:: ", step, "Accuracy:: ", self.feed_input(self.accuracy, sess, [new_batch, labels])

            self.feed_input(self.train_step, sess, [new_batch, labels])
            step+=1

    def train(self, sess, reader, saver):

        validation_sets = {}
        test_sets = {}
        for val_set in ['validation_synonym_3_5', 'validation_synonym_2_2', 'validation_synonym_1_1', 'validation_graph_3_5', 'validation_graph_2_2']:
            validation_sets[val_set] = reader.read_complete_set(val_set+ '_triplets', val_set + '_labels')
        for test_set in ['test_synonym_3_5', 'test_synonym_2_2', 'test_synonym_1_1', 'test_graph_3_5', 'test_graph_2_2']:
            test_sets[test_set] = reader.read_complete_set(test_set+ '_triplets', test_set + '_labels')
        all_training = reader.read_complete_set('training_triplets', 'training_labels')

        for epoch in range(20):
            reader.reset_reader()
            self.run_epoch(sess, reader, epoch, validation_sets, all_training)
            saver.save(sess, 'checkpoints/training.ckpt')

        for test_set in test_sets:
            print test_set, "Accuracy :: ", self.feed_input(self.accuracy, sess, test_sets[test_set])

def traain():
    modelConfig = ncr_cnn_model.bigConfig()
    trainConfig = firstTrainConfig()
    data = read_files()

    reader = DataReader(data)

    tr = TrainingUnit(modelConfig, trainConfig)
    saver = tf.train.Saver()

    init_op=tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    tr.train(sess, reader, saver)

    #	saver.restore(sess, 'checkpoints/training.ckpt')
    for test_set in ['test_synonym_3_5', 'test_synonym_2_2', 'test_synonym_1_1', 'test_graph_3_5', 'test_graph_2_2']:
        tmp_set = reader.read_complete_set(test_set+ '_triplets', test_set + '_labels')
        print test_set, "Accuracy :: ", tr.feed_input(tr.accuracy, sess, tmp_set)




def main():
    traain()
#	tesst()

if __name__ == "__main__":
    main()

