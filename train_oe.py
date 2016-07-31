import tensorflow as tf
from ordered_embeding import NCRModel
import process_querry_ordered
import reader
import argparse


class newConfig:
	batch_size = 4
	hpo_size = 10000
	comp_size = 400
	vocab_size = 50000
	stemmed_vocab_size = 50000
	hidden_size = 10
	word_embed_size = 100
	num_layers = 1
	max_sequence_length = 22
	alpha = 1
	beta = 1
	last_state = True

def run_epoch(sess, model, train_step, model_loss, rd, saver):
	rd.reset_counter()

	'''
	batch = rd.read_batch(50, newConfig.comp_size)
	batch_feed = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_hpo_id:batch[2]}
	#batch_feed = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_hpo_id:batch[2], model.input_comp:batch[3], model.input_comp_mask:batch[4]}
	#print sess.run(model.distances, feed_dict = batch_feed)[0].shape
	print "hello!"
	print sess.run(model.r_difs, feed_dict = batch_feed).shape
	print "bye!"
	print sess.run(model.p2c, feed_dict = batch_feed).shape
	print sess.run(model.c2c_neg, feed_dict = batch_feed).shape
	print sess.run(model.c2c_pos, feed_dict = batch_feed).shape
	print sess.run(model.new_loss, feed_dict = batch_feed).shape
	exit()
	print sess.run(model.densed_outputs, feed_dict = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_hpo_id:batch[2]})[0].shape
	print sess.run(model.diffs, feed_dict = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_hpo_id:batch[2]})[0].shape
	print sess.run(model.new_loss, feed_dict = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_hpo_id:batch[2]}).shape
	exit()
	'''

	ii = 0
	loss = 0
	report_len = 5
	while True:
		batch = rd.read_batch(newConfig.batch_size, newConfig.comp_size)
		if ii == 10 or batch == None:
			break
		batch_feed = {model.input_sequence : batch['seq'], model.input_stemmed_sequence : batch['stem_seq'], model.input_sequence_lengths: batch['seq_len'], model.input_hpo_id:batch['hp_id']}
		#batch_feed = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_hpo_id:batch[2], model.input_comp:batch[3], model.input_comp_mask:batch[4]}
		#batch_feed = {model.input_sequence : batch[0], model.input_sequence_lengths: batch[1], model.input_comp:batch[3], model.input_comp_mask:batch[4]}

		_ , step_loss = sess.run([train_step, model_loss], feed_dict = batch_feed)
		loss += step_loss

		if ii % report_len == report_len-1:
			print "Step =", ii+1, "\tLoss =", loss/report_len
			loss = 0
		ii += 1

def main():
	print "train_oe"
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--repdir', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="checkpoints/")
	args = parser.parse_args()

	oboFile = open("hp.obo")
	vectorFile = open("vectors.txt")
	stemmedVectorFile = open("stemmed_vectors.txt")

	rd = reader.Reader(oboFile, vectorFile, stemmedVectorFile)
#	rd.init_pmc_data(open('pmc_samples.p'),open('pmc_id2text.p'), open('pmc_labels.p'))
#	rd.init_wiki_data(open('wiki-samples.p'))
	
	newConfig.vocab_size = rd.word2vec.shape[0]
	newConfig.stemmed_vocab_size = rd.stemmed_word2vec.shape[0]
	newConfig.word_embed_size = rd.word2vec.shape[1]
	newConfig.max_sequence_length = rd.max_length
	newConfig.hpo_size = len(rd.concept2id)

	model = NCRModel(newConfig)
	model_loss = model.get_loss()

	lr = tf.Variable(0.01, trainable=False)
	train_step = tf.train.AdamOptimizer(lr).minimize(model_loss)

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	sess.run(tf.assign(model.word_embedding, rd.word2vec))
	sess.run(tf.assign(model.stemmed_word_embedding, rd.stemmed_word2vec))
	sess.run(tf.assign(model.ancestry_masks, rd.ancestry_mask))
	
	saver = tf.train.Saver()

	lr_init = 0.01
	lr_decay = 0.8

	samplesFile = open("labeled_data")
	ant =process_querry_ordered.NeuralAnnotator(model, rd, sess)
	samples = process_querry_ordered.prepare_samples(rd, samplesFile)

	training_samples = {}
	for hpid in rd.names:
		for s in rd.names[hpid]:
			training_samples[s]=[hpid]

	with open(args.repdir+"/test_results.txt","w") as testResultFile:
		testResultFile.write("")

	for epoch in range(100):
		print "epoch ::", epoch

		lr_new = lr_init * (lr_decay ** max(epoch-4.0, 0.0))
		sess.run(tf.assign(lr, lr_new))

		run_epoch(sess, model, train_step, model_loss, rd, saver)

		hit, total = ant.find_accuracy(samples, 5)
		print "Accuracy on test set ::", float(hit)/total
		with open(args.repdir+"/test_results.txt","a") as testResultFile:
			testResultFile.write(str(float(hit)/total)+"\n")

		hit, total = ant.find_accuracy(training_samples, 5)
		print "Accuracy on training set ::", float(hit)/total

#		saver.save(sess, args.repdir+'/training.ckpt') ## TODO

if __name__ == "__main__":
	main()

