class Config:
	batch_size = 128
	hidden_size = 1024

        layer1_size = 512
        layer2_size = 256
        layer3_size = 256

        dec_layer1_size = 256
        dec_layer2_size = 512
        dec_layer3_size = 1024

        z_dim = 16
        alpha = 0.2

	@staticmethod
	def update_with_reader(rd):
		#		Config.vocab_size = rd.word2vec.shape[0]
		Config.word_embed_size = 100 #rd.word2vec.shape[1]
		Config.max_sequence_length = int(rd.max_length)
		Config.hpo_size = len(rd.concepts)
#		Config.concept_NULL = rd.concept_NULL


