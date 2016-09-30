import create_sent_data
import cPickle as pickle

def process_sent(text):
	results = pickle.load(open('biolark_results.p', "rb"))
	return results[text]


def main():
	print process_sent('A syndrome of brachydactyly (absence of some middle or distal phalanges), aplastic or hypoplastic nails, symphalangism (ankylois of proximal interphalangeal joints), synostosis of some carpal and tarsal bones, craniosynostosis, and dysplastic hip joints is reported in five members of an Italian family.')

if __name__ == "__main__":
	main()
