class Config:
	batch_size = 32
	hidden_size = 300
	alpha = 1

	@staticmethod
	def update_with_reader(rd):
		Config.vocab_size = rd.word2vec.shape[0]
		Config.word_embed_size = rd.word2vec.shape[1]
		Config.max_sequence_length = rd.max_length
		Config.hpo_size = len(rd.concept2id)


