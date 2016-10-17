import tensorflow as tf
import argparse
import phraseConfig
import sent_model 
import sent_accuracy
import reader
import sent_annotator
import gpu_access
import h5py


def run_epoch(sess, model, train_step, model_loss, rd, saver, config):
	rd.reset_counter()

	'''
	batch = rd.read_batch(50, newConfig.comp_size)
	batch_feed = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_hpo_id:batch[2]}
	print sess.run(model.new_loss, feed_dict = batch_feed).shape
	exit()
	'''

	ii = 0
	loss = 0
	report_len = 20
	while True:
		batch = rd.read_batch(config.batch_size) #, config.comp_size)
		if ii == 200000 or batch == None:
			break
		batch_feed = {model.input_sequence : batch['seq'], model.input_sequence_lengths: batch['seq_len'], model.input_hpo_id:batch['hp_id']}

		_ , step_loss = sess.run([train_step, model_loss], feed_dict = batch_feed)
		loss += step_loss

		if ii % report_len == report_len-1:
			print "Step =", ii+1, "\tLoss =", loss/report_len
			loss = 0
		ii += 1

def train(repdir):
	print "Training..."

	oboFile = open("data/hp.obo")
	vectorFile = open("data/vectors.txt")

	print "Initializing Reader..."

	rd = reader.Reader(oboFile, vectorFile, addNull=True)

	print "Init pmc..."
	rd.init_pmc_data(open('data/pmc_samples.p'),open('data/pmc_id2text.p'), open('data/pmc_labels.p'))
	print "Init wiki..."
#	rd.init_wiki_data(open('data/wiki-samples.p'))
	print "reset counter..."
	rd.reset_counter()

	config = phraseConfig.Config
	config.update_with_reader(rd)

	print "Done.\n"

	print "Initializing Model..."
	
	model = sent_model.NCRModel(config, training=True)
	model_loss = model.loss

	lr_init = 0.01
	lr_decay = 0.8
	lr = tf.Variable(lr_init, trainable=False)
	train_step = tf.train.AdamOptimizer(lr).minimize(model_loss)

	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	sess.run(tf.initialize_all_variables())

	h5f = h5py.File('phrase_embeddings.h5', 'r')

	sess.run(tf.assign(model.word_embedding, h5f['word_embed'][:]))
	sess.run(tf.assign(model.HPO_embedding, h5f['hpo_embed'][:]))
#	sess.run(tf.assign(model.ancestry_masks, rd.ancestry_mask))
	
	print "Done.\n"

	saver = tf.train.Saver()


###	
	'''
	samplesFile = open("data/labeled_data")
	ant = annotator.NeuralPhraseAnnotator(model, rd, sess, False)
	samples = accuracy.prepare_phrase_samples(rd, samplesFile)
	'''
###	

	ant = sent_annotator.NeuralSentenceAnnotator(model, rd, sess, False)
###	
	'''
	training_samples = {}
	for hpid in rd.names:
		for s in rd.names[hpid]:
			training_samples[s]=[hpid]
	'''
###	

	with open(repdir+"/test_results.txt","w") as testResultFile:
		testResultFile.write("")

	for epoch in range(20):
		print "epoch ::", epoch

		lr_new = lr_init * (lr_decay ** max(epoch-4.0, 0.0))
		sess.run(tf.assign(lr, lr_new))

		run_epoch(sess, model, train_step, model_loss, rd, saver, config)
		text = open('../data/text/1003450').read()
		ant.process_text(text, 3.0)
#		sent_accuracy.find_sent_accuracy(lambda text: [x[0] for sent_res in ant.process_text(text, 1.0) for x in sent_res], '../data/', ant.rd)
		sent_accuracy.find_sent_accuracy(lambda text: [x[0] for sent_res in ant.process_text(text, 1.0) for x in sent_res], "labeled_sentences.p", ant.rd)
		'''
		hit, total = accuracy.find_phrase_accuracy(ant, samples, 5, False)
		print "Accuracy on test set ::", float(hit)/total
		with open(repdir+"/test_results.txt","a") as testResultFile:
			testResultFile.write(str(float(hit)/total)+"\n")
		'''
		
		'''
		hit, total = ant.find_accuracy(training_samples, 5)
		print "Accuracy on training set ::", float(hit)/total
		'''

	saver.save(sess, repdir+'/sentence_training.ckpt') ## TODO


def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--gpu', action='store_true', default=False)
	parser.add_argument('--repdir', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="sent_checkpoints/")
	args = parser.parse_args()
	if args.gpu:
		board = gpu_access.get_gpu()
		with tf.device('/gpu:'+board):
			train(args.repdir)
	else:
		with tf.device('/cpu:0'):
			train(args.repdir)

if __name__ == "__main__":
	main()

