import reader
import phrase_model
import accuracy
import annotator
import phraseConfig
import tensorflow as tf

def test_accuarcy_phrase():
	samplesFile = open("data/labeled_data")
	ant = annotator.create_annotator("checkpoints_backup/", "data/", True, False)
	samples = accuracy.prepare_phrase_samples(ant.rd, samplesFile)
	cor, tot = accuracy.find_phrase_accuracy(ant, samples, 5)
	print cor, tot
	print float(cor)/tot

def main():
	test_accuarcy_phrase()


if __name__ == "__main__":
	main()
