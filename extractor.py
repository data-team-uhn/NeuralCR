import tensorflow as tf
import numpy as np
import random
import fasttext
from onto import Ontology 


def tokenize(phrase):
	tmp = phrase.lower().replace(',',' , ').replace('-',' ').replace(';', ' ; ').replace('/', ' / ').replace('(', ' ( ').replace(')', ' ) ').replace('.', ' . ').strip().split()
	return ["INT" if w.isdigit() else ("FLOAT" if is_number(w) else w) for w in tmp]


def create_negatives(text, num):
    neg_tokens = tokenize(text)

    indecies = np.random.choice(len(neg_tokens), num)
    lengths = np.random.randint(1, 10, num)

    negative_phrases = [' '.join(neg_tokens[indecies[i]:indecies[i]+lengths[i]])
                                for i in range(num)]
    return negative_phrases


class ExtConfig:
    batch_size = 256
    hidden_size = 128
    layer1_size = 128
    layer2_size = 128
    layer3_size = 128
    max_sequence_length = 50
    lr = 1.0/512
    word_embed_size = 100

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


class ExtractModel():

    def phrase2vec(self, phrase_list, max_length):
        phrase_vec_list = []
        phrase_seq_lengths = []
        for phrase in phrase_list:
            tokens = tokenize(phrase)[:max_length-1]
            # TODO get the embedding size
            phrase_vec_list.append([self.word_model[tokens[i]] if i<len(tokens) else [0]*self.word_model.dim for i in range(max_length)])
            phrase_seq_lengths.append(len(tokens))
        return np.array(phrase_vec_list), np.array(phrase_seq_lengths)

    def encode(self, seq, seq_length):
        filters1 = tf.get_variable('conv1', [1, self.config.word_embed_size, self.config.word_embed_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        #conv1_b = tf.get_variable('conv1_b', initializer=tf.contrib.layers.xavier_initializer(), shape=self.config.hidden_size)
        conv_layer1 = tf.nn.conv1d(seq, filters1, 1, padding='SAME')
        with tf.variable_scope('fw'):
            cell_fw = tf.contrib.rnn.GRUCell(self.config.hidden_size/2, activation=tf.nn.tanh)
        with tf.variable_scope('bw'):
            cell_bw = tf.contrib.rnn.GRUCell(self.config.hidden_size/2, activation=tf.nn.tanh)
        _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, conv_layer1, dtype=tf.float32, sequence_length=seq_length)
        state = tf.concat(states, 1)

        layer1 = tf.nn.relu(linear('sm_layer1', state, [self.config.hidden_size, self.config.layer1_size]))
        layer2 = tf.nn.relu(linear('sm_layer2', layer1, [self.config.layer1_size, self.config.layer2_size]))
        layer3 = tf.nn.relu(linear('sm_layer3', layer2, [self.config.layer2_size, self.config.layer3_size]))
        return (linear('sm_layer4', layer3, [self.config.layer3_size, 1]))

    '''
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
    '''
 
    #############################
    ##### Creates the model #####
    #############################
    def __init__(self, config, word_model):
        tf.reset_default_graph()
        self.word_model = word_model
        self.config = config
        self.label = tf.placeholder(tf.float32, shape=[None,1])
        self.seq = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, config.word_embed_size])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])

        self.seq_logit = self.encode(self.seq, self.seq_len)
        self.pred = tf.nn.sigmoid(self.seq_logit)
        self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.label, self.seq_logit)) 

	self.lr = tf.Variable(config.lr, trainable=False)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	self.sess.run(tf.global_variables_initializer())

    #########################################################

    def predict(self, querries):
        seq, seq_len = self.phrase2vec(querries, self.config.max_sequence_length)
        querry_dict = {self.seq : seq, self.seq_len: seq_len}
        return self.sess.run(self.pred, feed_dict = querry_dict)

    def save_params(self, repdir='.'):
        tf.train.Saver().save(self.sess, (repdir+'/params.ckpt').replace('//','/'))

    def load_params(self, repdir='.'):
        tf.train.Saver().restore(self.sess, (repdir+'/params.ckpt').replace('//','/'))

    def fit(self, pos_samples, neg_samples):
        negs = {}
        poses = {}
        negs['seq'], negs['seq_len'] = self.phrase2vec(neg_samples, self.config.max_sequence_length)
        poses['seq'], poses['seq_len'] = self.phrase2vec(pos_samples, self.config.max_sequence_length)

        
        for epoch in range(30):
            print "Epoch::", epoch
            all_samples = {}
            all_samples['seq'] = np.concatenate([poses['seq'], negs['seq']])
            all_samples['seq_len'] = np.concatenate([poses['seq_len'],negs['seq_len']])
            all_samples['label'] = np.array([[1.0]]*len(pos_samples)+[[0.0]]*len(neg_samples))
            training_size = all_samples['seq'].shape[0]
            shuffled_indecies = range(training_size)
            random.shuffle(shuffled_indecies)

            ct = 0
            report_loss = 0
            total_loss = 0
            report_len = 20
            head = 0
            if epoch%5 == 0:
                print self.predict(['retina cancer', 'cancer', 'kidney neoplasm', 'retina', 'house', 'paitient', 'kidney'])
            if epoch%10 == 0 and epoch>0:
                self.save_params('ext_params/')

            while head < training_size:
                ending = min(training_size, head + self.config.batch_size)
                batch = {}
                for cat in all_samples:
                    batch[cat] = all_samples[cat][shuffled_indecies[head:ending]]
                head += self.config.batch_size
                batch_feed = {self.seq:batch['seq'],\
                        self.seq_len:batch['seq_len'],\
                        self.label:batch['label']} 
                _ , batch_loss = self.sess.run([self.train_step, self.loss], feed_dict = batch_feed)

                report_loss += batch_loss
                total_loss += batch_loss
                if ct % report_len == report_len-1:
                    print "Step =", ct+1, "\tLoss =", report_loss/report_len
                    #sys.stdout.flush()
                    report_loss = 0
                ct += 1

def main():
    word_model = fasttext.load_model('data/model_pmc.bin')
    hpo = Ontology('data/hp.obo',"HP:0000118")

    wiki_text = open('data/wiki_text').read()
    wiki_negs = create_negatives(wiki_text[:10000000], 10000)

    ubs = [Ontology('data/uberon.obo', root) for root in ["UBERON:0000062", "UBERON:0000064"]]
    neg = set([name for ub in ubs for concept in ub.names for name in ub.names[concept]] + wiki_negs)
    pos = set([name for concept in hpo.names for name in hpo.names[concept]])

    model = ExtractModel(ExtConfig(), word_model)
    model.load_params('ext_params/')
    print model.predict(['retina','retina cancer', 'chromosome'])
#    model.fit(pos,neg)
    
if __name__ == '__main__':
    main()
