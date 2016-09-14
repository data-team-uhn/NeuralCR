import argparse
import os
import reader

def parse_results(address, rd):
	res={}
	for f in os.listdir(address):
		res[f]=[rd.real_id[x.replace("_",":")] for x in open(address+"/"+f).read().strip().split("\n") if len(x)>0]
	return res

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('ground_truth')
	parser.add_argument('method_res')
#    parser.add_argument('method_res_other')
	args = parser.parse_args()
	oboFile = open("hp.obo")
	vectorFile = open("vectors.txt")
	samplesFile = open("labeled_data")
	stemmedVectorFile = open("stemmed_vectors.txt")
	rd = reader.Reader(oboFile, vectorFile, stemmedVectorFile)

	method_res=parse_results(args.method_res, rd)
#    method_res_other=parse_results(args.method_res_other)
	ground_truth=parse_results(args.ground_truth, rd)


	positives=0
	true_positives=0
	calls=0
	for text in ground_truth:
		true_positives+=len([x for x in method_res[text] if x in ground_truth[text]])
		missed_ones = [x for x in ground_truth[text] if x not in method_res[text]]
#        interesting_ones = [x for x in ground_truth[text] if x not in method_res[text] and x in method_res_other[text]]
		calls+=len(method_res[text])
		positives+=len(ground_truth[text])
		#print text, missed_ones
		#print text, interesting_ones

	print "Sensitivity :: ", float(true_positives)/positives
	print "Precision :: ", float(true_positives)/calls

if __name__ == "__main__":
	main()

