class Config:
    include_negs = True
    batch_size = 256
    cl1 = 2048
    cl2 = 2048

    max_sequence_length = 50
    lr = 1.0/512

    word_embed_size = 100
    @staticmethod
    def update_with_reader(ont):
        Config.concepts_size = len(ont.concepts)+1




