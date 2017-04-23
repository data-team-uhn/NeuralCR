class Config:
    batch_size = 256
    hidden_size = 512

    layer1_size = 1024
    layer2_size = 1024
    layer3_size = 1024
    layer4_size = 1024
    #layer4_size = 2048
    lr = 1.0/512
#    lr = 1.0/1024
#    lr = 0.002

    word_embed_size = 100
    @staticmethod
    def update_with_reader(rd):
        Config.max_sequence_length = int(rd.max_length)
        Config.hpo_size = len(rd.concepts)




