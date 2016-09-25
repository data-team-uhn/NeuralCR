import reader
import phrase_model
import accuracy
import annotator
import phraseConfig
import argparse
import tensorflow as tf
import gpu_access

def test_accuarcy_phrase():
	samplesFile = open("data/labeled_data")
	ant = annotator.create_annotator("checkpoints/", "data/", True)
	samples = accuracy.prepare_phrase_samples(ant.rd, samplesFile)
	cor, tot = accuracy.find_phrase_accuracy(ant, samples, 1)
	print float(cor)/tot

def main():
	parser = argparse.ArgumentParser(description='Hello!')
	parser.add_argument('--gpu', action='store_true', default=False)
	args = parser.parse_args()


	if args.gpu:
		board = gpu_access.get_gpu()
		with tf.device('/gpu:'+board):
			test_accuarcy_phrase()
	else:
		with tf.device('/cpu:0'):
			test_accuarcy_phrase()

if __name__ == "__main__":
	main()
