import os

def prepare_phrase_samples(rd, samplesFile):
	samples = {}
	ct = 0
	for line in samplesFile:
		tokens = line.strip().split("\t")
		real_hp_id = rd.real_id[tokens[1].strip().replace("_",":")]
		if real_hp_id not in rd.concepts:
			continue
		samples[tokens[0].strip()] = [real_hp_id]
		ct += 1
		if ct == 500000:
			break
	return samples


#################### Accuracy function ####################

def parse_results(address, rd):
	res={}
	for f in os.listdir(address):
		res[f]=[rd.real_id[x.replace("_",":")] for x in open(address+"/"+f).read().strip().split("\n") if len(x)>0]
	return res


def find_sent_accuracy(text_predictor, address, rd, verbose=False):
	files =os.listdir(address+"/text")

	positives=0
	true_positives=0
	calls=0

	counter = 0

	for f in files:
		text = open(address+"/text/"+f).read()
#		ground_truth = [rd.real_id[x.replace("_",":")] for x in open(address+"/ground_truth/"+f).read().strip().split("\n") if len(x)>0]
		ground_truth = set([rd.real_id[x.replace("_",":")] for x in open(address+"/ground_truth/"+f).read().strip().split("\n") if len(x)>0])
		#missed_ones = [x for x in ground_truth if x not in method_results]
		method_results = set(text_predictor(text))

		true_positives+=len([x for x in method_results if x in ground_truth])
		positives+=len(ground_truth)
		calls += len(method_results)
		'''
		if counter % 10 == 0:
			print str(100.0*counter/len(files)) + '%'
			print "Sensitivity :: ", float(true_positives)/positives
			print "Precision :: ", float(true_positives)/calls
		counter += 1
		'''
	
#	print true_positives, positives, calls
	print "Sensitivity :: ", float(true_positives)/positives
	if calls>0:
		print "Precision :: ", float(true_positives)/calls
	else:
		print "No calls!"
	return 

def main():
	find_sent_accuracy(None, '../data/')


if __name__ == '__main__':
	main()
