import reader
import phrase_model
import accuracy
import annotator
import phraseConfig

def main():
	ant = annotator.create_annotator("checkpoints/", "data/", False)

	samplesFile = open("data/labeled_data")
	samples = accuracy.prepare_phrase_samples(ant.rd, samplesFile)

	cor, tot = accuracy.find_phrase_accuracy(ant, samples, 5)
	print float(cor)/tot



if __name__ == "__main__":
	main()
