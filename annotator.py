
class NeuralAnnotator:

	def get_hp_id(self, querry, count=5):
		inp = self.rd.create_test_sample(querry)
		querry_dict = {self.model.input_sequence : inp['seq'], self.model.input_stemmed_sequence : inp['stem_seq'], self.model.input_sequence_lengths: inp['seq_len']}
		res_querry = self.sess.run(self.querry_distances, feed_dict = querry_dict)
		results=[]
		for s in range(len(querry)):
			indecies_querry = np.argsort(res_querry[s,:])
			tmp_res = []
			num_printed = 0
			for i in indecies_querry:
				num_printed += 1
				tmp_res.append((self.rd.concepts[i],res_querry[s,i]))
				if num_printed>=count:
					break
			results.append(tmp_res)
		return results

	'''
	def get_hp_id_comp_phrase(self, querry, count=5):
		inp = self.rd.create_test_sample(querry)
		querry_dict = {self.model.input_sequence : inp['seq'], self.model.input_stemmed_sequence : inp['stem_seq'], self.model.input_sequence_lengths: inp['seq_len']}
		res_querry = self.sess.run(self.querry_distances_phrases, feed_dict = querry_dict)
		results=[]
		for s in range(len(querry)):
			indecies_querry = np.argsort(res_querry[s,:])
			tmp_res = []
			for i in indecies_querry:
				res_item = (self.rd.concepts[self.rd.name2conceptid.values()[i]],res_querry[s,i])
				if res_item[0] not in [x[0] for x in tmp_res]:
					tmp_res.append(res_item)
				if len(tmp_res)>=count:
					break
			results.append(tmp_res)
		return results
	'''

	def __init__(self, model, rd ,sess, compWithPhrases = False):
		self.model=model
		self.rd = rd
		self.sess = sess
		self.compWithPhrases = compWithPhrases

		if self.compWithPhrases:
			inp = self.rd.create_test_sample(self.rd.name2conceptid.keys())
			querry_dict = {self.model.input_sequence : inp['seq'], self.model.input_stemmed_sequence : inp['stem_seq'], self.model.input_sequence_lengths: inp['seq_len']}
			res_querry = self.sess.run(self.model.gru_state, feed_dict = querry_dict)
			ref_vecs = tf.Variable(res_querry, False)
			sess.run(tf.assign(ref_vecs, res_querry))
			self.querry_distances = self.model.euclid_dis_cartesian(ref_vecs, self.model.gru_state)
		else:
			self.querry_distances = self.model.get_querry_dis()

