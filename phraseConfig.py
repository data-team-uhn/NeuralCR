class Config:
	batch_size = 128
	hidden_size = 200
	concept_size = 200
	alpha = 1
	n_types = 2
	l1_size = 100
	l2_size = 100

	@staticmethod
	def update_with_reader(rd):
		#		Config.vocab_size = rd.word2vec.shape[0]
		Config.word_embed_size = 100 #rd.word2vec.shape[1]
		Config.max_sequence_length = rd.max_length
		Config.hpo_size = len(rd.concepts)
#		Config.concept_NULL = rd.concept_NULL


