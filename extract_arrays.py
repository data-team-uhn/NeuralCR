import tensorflow as tf
import phrase_model
import phraseConfig
import reader
import argparse
import h5py
import sys
import cPickle as pickle

def extract_raw_word_vectors():
	vectorFile = open("data/vectors.txt")
	oboFile = open("data/hp.obo")
	rd = reader.Reader(oboFile, vectorFile)
	
	h5f = h5py.File('data/word_vectors.h5', 'w')
	h5f.create_dataset('word_embed', data=rd.word2vec)
	h5f.close()
	pickle.dump(rd.word2id, open('data/word2id.p', "wb"))

def extract(repdir):
	oboFile = open("data/hp.obo")
	vectorFile = open("data/vectors.txt")

	rd = reader.Reader(oboFile, vectorFile)
	config = phraseConfig.Config
	config.update_with_reader(rd)

	model = phrase_model.NCRModel(config)

	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	saver = tf.train.Saver()
	saver.restore(sess, repdir + '/training.ckpt')

	HPO_embed = sess.run(model.HPO_embedding)
	word_embed = sess.run(model.word_embedding)

	if len(rd.word2id) != word_embed.shape[0]:
		sys.stderr.write("Error! There is something wrong with the number of word vectors.\n")
		exit()
	h5f = h5py.File('phrase_embeddings.h5', 'w')
	h5f.create_dataset('hpo_embed', data=HPO_embed)
	h5f.create_dataset('word_embed', data=word_embed)
	h5f.close()

def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--gpu', action='store_true', default=False)
	parser.add_argument('--repdir', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="checkpoints/")
	args = parser.parse_args()
	extract_raw_word_vectors()
	return

	'''
	h5f = h5py.File('phrase_embeddings.h5', 'r')
	hpo = h5f['hpo_embed'][:]
	word = h5f['word_embed'][:]
	print hpo
	print hpo.shape
	print word.shape
	h5f.close()
	exit()
	'''


	if args.gpu:
		board = gpu_access.get_gpu()
		with tf.device('/gpu:'+board):
			extract(args.repdir)
	else:
		with tf.device('/cpu:0'):
			extract(args.repdir)



if __name__ == "__main__":
	main()
