
def prepare_phrase_samples(rd, samplesFile, filter_flag=False):
	samples = {}
	ct = 0
	for line in samplesFile:
		tokens = line.strip().split("\t")
                if len(tokens)>2:
                    continue
		real_hp_id = rd.real_id[tokens[1].strip().replace("_",":")]
		if real_hp_id not in rd.concepts:
			continue
		samples[tokens[0].strip()] = [real_hp_id]
		ct += 1
		if ct == 500000:
			break
        if filter_flag:
            copy_samples = dict(samples)
            for x in copy_samples:
                normed = x.lower()
                if normed in rd.name2conceptid:
                    del samples[x]

	return samples


#################### Accuracy function ####################

def find_phrase_accuracy(ant, samples, top_size=None, verbose=False):
	header = 0
	batch_size = 512

	hit=[0]*top_size
	nohit=0

	while header < len(samples):
		batch = {x:samples[x] for x in samples.keys()[header:min(header+batch_size, len(samples))]}
		header += batch_size
		results = ant.get_hp_id(batch.keys(), top_size)
		#results = ant.get_hp_id(batch.keys(), 50000)
		#results = self.get_hp_id(batch.keys(),top_size)
		for i,s in enumerate(batch):
			hashit = False
			for attempt,res in enumerate(results[i]):
                                if attempt >= top_size:
                                    break
				if res[0] in batch[s]:
					hit[attempt] += 1
					hashit = True
					break
			if not hashit:
				nohit += 1
				if verbose:
					print "\n---------------------"
					print "---------------------"
					print "---------------------\n"
                                        printed_sth = False
					for attempt, res in enumerate(results[i]):
                                            if res[0] == batch[s][0]:
                                                print s, batch[s], ant.rd.names[batch[s][0]], "#"+str(attempt), res[1]
                                                printed_sth = True
                                                break
                                        if not printed_sth:
                                            print s, batch[s], ant.rd.names[batch[s][0]]
					print ""
					for attempt,res in enumerate(results[i]):
                                            if attempt >= top_size:
                                                break
                                            if res[0] in ant.rd.names:
                                                print res[0], ant.rd.names[res[0]], res[1]
                                            else:
                                                print res[0], "--not found--", res[1]

	total = sum(hit) + nohit
	return total - nohit, total

