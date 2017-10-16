import phrase_model 
import accuracy
import numpy as np
from onto import Ontology
import fasttext

def main():
    print "Loading word model" 
    word_model = fasttext.load_model('data/model_pmc.bin')
    print "Loading ontology" 
    ont = Ontology('data/hp.obo',"HP:0000118")

    config = phrase_model.Config
    model = phrase_model.NCRModel(config, ont, word_model)

    samplesFile = open("data/labeled_data")
    samples = accuracy.prepare_phrase_samples(model.ont, samplesFile, True)
    #'''
    model.init_training()
    num_epochs = 40
    for epoch in range(num_epochs):
        print "Epoch::", epoch
        model.train_epoch(verbose=False)
        
        if epoch>0 and (epoch%10 ==0 or epoch==num_epochs-1):
            hit, total = accuracy.find_phrase_accuracy(model, samples, 5, False)
            print "R@5 Accuracy on val set ::", float(hit)/total

            hit, total = accuracy.find_phrase_accuracy(model, samples, 1, False)
            print "R@1 Accuracy on val set ::", float(hit)/total
    model.save_params('exp_params')
    #'''


    '''
    model.load_params('exp_params')
    embeddings = model.sess.run(model.embeddings)
    '''
    #child = embeddings[ont.concept2id['HP:0012777']]
    #parent = embeddings[ont.concept2id['HP:0000479']]
    #print parent - child
  # values = [x for row in embeddings for x in row]



    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
#    sns.heatmap(parent-child)
    sns.distplot(parent-child)
    #sns.distplot(np.array(values))
    plt.show()
    '''


if __name__ == '__main__':
    main()
