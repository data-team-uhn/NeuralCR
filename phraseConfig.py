class Config:
	batch_size = 128
	hidden_size = 1024

        layer1_size = 2048
        layer2_size = 2048
        layer3_size = 800

	@staticmethod
	def update_with_reader(rd):
		#		Config.vocab_size = rd.word2vec.shape[0]
		Config.word_embed_size = 100 #rd.word2vec.shape[1]
		Config.max_sequence_length = rd.max_length
		Config.hpo_size = len(rd.concepts)
#		Config.concept_NULL = rd.concept_NULL


