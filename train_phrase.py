import tensorflow as tf
import argparse
import phraseConfig
import phrase_model 
import accuracy
#import reader
import fasttext_reader as reader
import phrase_annotator
import gpu_access
import numpy as np
import sys
import h5py


def run_epoch(sess, model, train_step, model_loss, rd, config):
	rd.reset_counter()
	rd.reset_counter_by_concept()
        


        '''
	batch = rd.read_batch(5)
        batch_feed = {model.input_sequence : batch['seq'], model.input_sequence_lengths: batch['seq_len'], model.input_hpo_id:batch['hp_id']}
        print sess.run(model.z, feed_dict = batch_feed)[:1]
	exit()
        '''
	ii = 0
	loss = 0
	report_len = 20
	while True:
		batch = rd.read_batch(config.batch_size) #, config.comp_size)
		#batch = rd.read_batch_by_concept(config.batch_size) #, config.comp_size)
		if ii == 10000000 or batch == None:
			break
		#print np.array(batch['hp_id']).T[0]
		batch_feed = {model.input_sequence : batch['seq'], model.input_sequence_lengths: batch['seq_len'], model.input_hpo_id:batch['hp_id'], model.phase:True} #, model.input_hpo_id_unique:batch['hp_id']} #, model.set_loss_for_input:True, model.set_loss_for_def:False}
		#batch_feed = {model.input_sequence : batch['seq'], model.input_sequence_lengths: batch['seq_len'], model.input_hpo_id:batch['hp_id'], model.input_hpo_id_unique:np.array(list(set(batch['hp_id'])))} #, model.set_loss_for_input:True, model.set_loss_for_def:False}

		'''
		print sess.run(model.first_vec, feed_dict = batch_feed)
		#print sess.run(model.cond, feed_dict = batch_feed)
		exit()
		'''

		_ , step_loss = sess.run([train_step, model_loss], feed_dict = batch_feed)
		#print step_loss
		#exit()
		loss += step_loss

		if ii % report_len == report_len-1:
			print "Step =", ii+1, "\tLoss =", loss/report_len
			sys.stdout.flush()
			
			loss = 0
		ii += 1

def train(repdir, lr_init, lr_decay, config, use_sparse_matrix=True):
	print "Training..."

	oboFile = open("data/hp.obo")
	vectorFile = open("data/vectors.txt")
	tf.reset_default_graph()
	rd = reader.Reader(oboFile) #, vectorFile)
	print "reader inited"
	#rd.init_uberon_list()
	config.update_with_reader(rd)
	
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        if use_sparse_matrix:
            model = phrase_model.NCRModel(config, training=True, ancs_sparse = rd.sparse_ancestrs)
        else:
            model = phrase_model.NCRModel(config, training=True)

	model_loss = model.loss

	lr = tf.Variable(0.02, trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(lr).minimize(model_loss)

	h5f = h5py.File('plot_data.h5', 'r')
	z = np.array(h5f['z'])

	sess.run(tf.initialize_all_variables())
        sess.run(tf.assign(model.z, z))
        if not use_sparse_matrix:
            sess.run(tf.assign(model.ancestry_masks, rd.ancestry_mask))
	##C
#	sess.run(tf.assign(model.word_embedding, rd.word2vec))
#	sess.run(tf.assign(model.descendancy_masks, rd.ancestry_mask.T))
	
	saver = tf.train.Saver()
	##C
#	saver.restore(sess, "/ais/gobi4/arbabi/codes/NeuralCR/checkpoints/training.ckpt") ## TODO

	samplesFile = open("data/labeled_data")
	ant = phrase_annotator.NeuralPhraseAnnotator(model, rd, sess, False)
	samples = accuracy.prepare_phrase_samples(rd, samplesFile, True)

	training_samples = {}
	for hpid in rd.names:
		for s in rd.names[hpid]:
			training_samples[s]=[hpid]
	
	##C
	with open(repdir+"/test_results.txt","w") as testResultFile:
		testResultFile.write("")

	##C
        for epoch in range(40):#, 40):
		print "epoch ::", epoch

		lr_new = lr_init * (lr_decay ** max(epoch-4.0, 0.0))
		sess.run(tf.assign(lr, lr_new))

		run_epoch(sess, model, train_step, model_loss, rd, config)
		for x in ant.get_hp_id(['retina cancer'], 10)[0]:
		#for x in ant.get_hp_id(['skeletal anomalies'], 10)[0]:
			print rd.names[x[0]], x[1]
		if False and (epoch % 5 == 0):
			saver.save(sess, repdir + '/training.ckpt') ## TODO
                if ((epoch>0 and epoch % 5 == 0)): # or (epoch > 25)):
			hit, total = accuracy.find_phrase_accuracy(ant, samples, 5, False)
			print "R@5 Accuracy on test set ::", float(hit)/total
                        hit, total = accuracy.find_phrase_accuracy(ant, samples, 1, False)
                        print "R@1 Accuracy on test set ::", float(hit)/total
#		with open(repdir+"/test_results.txt","a") as testResultFile:
#			testResultFile.write(str(float(hit)/total)+"\n")
		
		'''
		hit, total = ant.find_accuracy(training_samples, 5)
		print "Accuracy on training set ::", float(hit)/total
		'''

	hit_5, total_5 = accuracy.find_phrase_accuracy(ant, samples, 5, False)
        print "R@5 Accuracy on test set ::", float(hit_5)/total_5
	hit_1, total_1 = accuracy.find_phrase_accuracy(ant, samples, 1, False)
        print "R@1 Accuracy on test set ::", float(hit_1)/total_1

	saver.save(sess, repdir + '/training.ckpt') ## TODO
	return float(hit_5)/total_5, float(hit_1)/total_1


def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--repdir', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="checkpoints/")
	args = parser.parse_args()

	lr_init = 0.0005
	lr_decay = 0.95

	config = phraseConfig.Config
	config.batch_size = 128
	'''
        for config.batch_size in [128, 256]:
            for lr_init in [0.0005, 0.0002, 0.001]:
                for config.hidden_size in [512, 1024]:
                    for config.layer1_size in [config.hidden_size, 2*config.hidden_size]:
                        for config.layer2_size in [config.hidden_size, 2*config.hidden_size]:
                            for config.alpha in [0.1, 0.2, 0.3, 0.5]:
                                print "hi"
                                accuracy = train(args.repdir, lr_init, lr_decay, config)
                                with open("grid_results.txt","a") as testResultFile:
                                    testResultFile.write('lr_init: ' + str(lr_init) +\
                                                '\tlr_decay: ' + str(lr_decay) +\
                                                '\tbatch_size ' + str(config.batch_size) +\
                                                '\thidden_size ' + str(config.hidden_size) +\
                                                '\tlayer1_size ' + str(config.layer1_size) +\
                                                '\tlayer2_size ' + str(config.layer2_size) +\
                                                '\talpha ' + str(config.alpha) +\
                                                '\taccuracy: '+ str(accuracy) +"\n")
        return
	'''
	'''
	for config.l1_size in [100, 200, 300]:
		for config.l2_size in [100, 200, 300]:
			for config.hidden_size in [200, 300, 400, 100]:
				config.concept_size = config.hidden_size
				#with tf.device('/gpu:'+board):
				accuracy = train(args.repdir, lr_init, lr_decay, config)
				with open("grid_results_mlp.txt","a") as testResultFile:
					testResultFile.write('l1_size: ' + str(config.l1_size) +\
							'\tl2_size: ' + str(config.l2_size) +\
							'\thidden_size: ' + str(config.hidden_size) +\
							'\taccuracy: '+ str(accuracy) +"\n")
	'''
	'''
	train(args.repdir, lr_init, lr_decay, config)
	return
	'''
        train(args.repdir, lr_init, lr_decay, config)

if __name__ == "__main__":
	main()

