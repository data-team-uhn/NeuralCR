import os
import cPickle as pickle
import biolark_wrapper
import sent_level

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


#def find_sent_accuracy(biolark, text_predictor, labeled_data_file, rd, verbose=False):
def compare_methods(ref, challenger, labeled_data_file, rd):
	positives=0
	true_positives=0
	calls=0

	counter = 0
        
	data = pickle.load(open(labeled_data_file,'rb'))
	for sent in data:
		ref_results = set(ref(sent))
		challenger_results = set(challenger(sent))
		#biolark_results = set(biolark(sent))

		true_positives+=len([x for x in challenger_results if x in data[sent]])
		positives+=len(data[sent])
		calls += len(challenger_results)
		#fp = [x for x in method_results if x not in data[sent]]
		#fn = [x for x in data[sent] if x not in method_results]
		fp = [x for x in challenger_results if x not in data[sent] and x not in ref_results]
		fn = [x for x in data[sent] if x not in challenger_results and x in ref_results]
		if len(fp)+len(fn)>0:
			print "==============================="
			print sent
			print "True labels:"
			print "False positives:"
			for c in fp:
					print c, rd.names[c]
			print "False negatives:"
			for c in fn:
					print c, rd.names[c]

		if counter % 100 == 0 and counter>0:
			print str(100.0*counter/len(data.keys())) + '%'
			print "Sensitivity :: ", float(true_positives)/positives
			print "Precision :: ", float(true_positives)/calls
			return
		counter += 1

	print "Sensitivity :: ", float(true_positives)/positives
	if calls>0:
		print "Precision :: ", float(true_positives)/calls
	else:
		print "No calls!"
	return 



def find_sent_accuracy(text_predictor, labeled_data_file, rd, verbose=False):
	positives=0
	true_positives=0
	calls=0

	counter = 0

        print "finding accuracy"
	data = pickle.load(open(labeled_data_file,'rb'))
        print "finding accuracy"
	for sent in data:
		method_results = set(text_predictor(sent))
		#biolark_results = set(biolark(sent))

		true_positives+=len([x for x in method_results if x in data[sent]])
		positives+=len(data[sent])
		calls += len(method_results)
		#fp = [x for x in method_results if x not in data[sent]]
		#fn = [x for x in data[sent] if x not in method_results]
		'''
		fp = [x for x in method_results if x not in data[sent] and x not in biolark_results]
		fn = [x for x in data[sent] if x not in method_results and x in biolark_results]
		if len(fp)+len(fn)>0:
			print "==============================="
			print sent
			print "True labels:"
			print "False positives:"
			for c in fp:
					print c, rd.names[c]
			print "False negatives:"
			for c in fn:
					print c, rd.names[c]
		'''

		if counter % 100 == 0 and counter>0:
			print str(100.0*counter/len(data.keys())) + '%'
			print "Sensitivity :: ", float(true_positives)/positives
			print "Precision :: ", float(true_positives)/calls
			#return
		counter += 1

	print "Sensitivity :: ", float(true_positives)/positives
	if calls>0:
		print "Precision :: ", float(true_positives)/calls
	else:
		print "No calls!"
	return 



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

	'''
	textAnt = sent_annotator.Sent_ant_wrapper('sent_checkpoints_backup/', False)
	#sent_ant_func = lambda text: [x[0] for sent_res in textAnt.process_text(text, 1.0) for x in sent_res]
	find_sent_accuracy(sent_ant_func, "labeled_sentences.p", textAnt.ant.rd)
	find_sent_accuracy(biolark_wrapper.process_sent, "labeled_sentences.p", textAnt.ant.rd)
	return
	'''


	textAnt = sent_level.TextAnnotator("checkpoints", datadir="data/", addNull=True)
	#textAnt = sent_level.TextAnnotator("checkpoints_backup/", "data/", False)
	sent_window_func = lambda text: [x[2] for x in textAnt.process_text(text, -1.0, True )]
	compare_methods(biolark_wrapper.process_sent, sent_window_func, "labeled_sentences.p", textAnt.rd)
	#find_sent_accuracy(biolark_wrapper.process_sent, "labeled_sentences.p", textAnt.ant.rd)
	#find_sent_accuracy(sent_window_func, "labeled_sentences.p", textAnt.rd)

	#compare_methods(sent_ant_func, biolark_wrapper.process_sent, "labeled_sentences.p", textAnt.ant.rd)
	#compare_methods(biolark_wrapper.process_sent, sent_ant_func, "labeled_sentences.p", textAnt.ant.rd)
	return
	'''
	text = "Parental transmission of a structurally or functionally unbalanced chromosome complement can lead to 15q11-q13 deletions or to UPD and will result in case-specific recurrence risks."
	text = "Branchiootorenal (BOR) syndrome is a common autosomal dominant form of hearing impairment previously mapped to 8q."
	print biolark_wrapper.process_sent(text)
	return
	'''
	#textAnt = sent_annotator.Sent_ant_wrapper('sent_checkpoints_backup/', False)
	textAnt = sent_level.TextAnnotator("checkpoints_backup/", "data/", False)
	sent_window_func = lambda text: [x[2] for x in textAnt.process_text(text, 0.7, True )]
	find_sent_accuracy(sent_window_func, "labeled_sentences.p", textAnt.ant.rd)
	#compare_methods(biolark_wrapper.process_sent, sent_window_func, "labeled_sentences.p", textAnt.ant.rd)
	return

	#	find_sent_accuracy(biolark_wrapper.process_sent, "labeled_sentences.p")

#	textAnt = sent_level.TextAnnotator("checkpoints/", "data/")
	#textAnt = sent_level.TextAnnotator("checkpoints_backup/", "data/")
	#find_sent_accuracy(biolark_wrapper.process_sent, lambda text: [x[2] for x in textAnt.process_text(text, 1.0, True )], "labeled_sentences.p", textAnt.ant.rd)

	textAnt = sent_level.TextAnnotator("checkpoints_backup/", "data/", False)
	#textAnt = sent_level.TextAnnotator("checkpoints_backup/", "data/")
#	find_sent_accuracy(lambda text: [x[2] for x in textAnt.process_text(text, 0.5, True )], "labeled_sentences.p", textAnt.ant.rd)
	compare_methods(lambda text: [x[2] for x in textAnt.process_text(text, 0.5, True )], biolark_wrapper.process_sent, "labeled_sentences.p", textAnt.ant.rd)
	#compare_methods(biolark_wrapper.process_sent, lambda text: [x[2] for x in textAnt.process_text(text, 0.5, True )], "labeled_sentences.p", textAnt.ant.rd)
#	textAnt = sent_level.TextAnnotator("checkpoints_backup/", "data/")
#	find_sent_accuracy(biolark_wrapper.process_sent, "labeled_sentences.p", textAnt.ant.rd)
	return
	###





if __name__ == '__main__':
	main()
