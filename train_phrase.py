import argparse
import phraseConfig
import phrase_model 
import accuracy
import fasttext_reader as reader
import numpy as np

def new_train(config):
    report_len = 20
    num_epochs = 25 

    rd = reader.Reader(open("data/hp.obo"), False)
    model = phrase_model.NCRModel(config, rd)

    samplesFile = open("data/labeled_data")
    samples = accuracy.prepare_phrase_samples(rd, samplesFile, True)
    training_samples = {}
    for hpid in rd.names:
        for s in rd.names[hpid]:
            training_samples[s]=[hpid]
    
    for epoch in range(num_epochs):
        print "epoch ::", epoch
        model.train_epoch()
        for x in model.get_hp_id(['retina cancer'], 10)[0]:
            print rd.names[x[0]], x[1]
        #for x in ant.get_hp_id(['skeletal anomalies'], 10)[0]:
        if ((epoch>0 and epoch % 5 == 0)) or epoch == num_epochs-1:
            hit, total = accuracy.find_phrase_accuracy(model, samples, 5, False)
            print "R@5 Accuracy on test set ::", float(hit)/total
            hit, total = accuracy.find_phrase_accuracy(model, samples, 1, False)
            print "R@1 Accuracy on test set ::", float(hit)/total
        if ((epoch>0 and epoch % 10 == 0)) or epoch == num_epochs-1:
            hit, total = accuracy.find_phrase_accuracy(model, training_samples, 1, False)
            print "Accuracy on training set ::", float(hit)/total
    return model

def grid_search():
    config = phraseConfig.Config
    rd = reader.Reader(open("data/hp.obo"), False)
    config.update_with_reader(rd)

    samplesFile = open("data/labeled_data")
    samples = accuracy.prepare_phrase_samples(rd, samplesFile, True)

    training_samples = {}
    for hpid in rd.names:
        for s in rd.names[hpid]:
            training_samples[s]=[hpid]
    
    for config.batch_size in [128, 256, 512]:
        for config.lr in [0.0005, 0.0002, 0.0001]:
            for config.hidden_size in [512]:
                for config.layer1_size in [1024]:
                    for config.layer2_size in [1024, 2048]:
                        config.layer3_size = config.layer1_size

                        model = phrase_model.NCRModel(config, rd)
                        num_epochs = 20 
                        for epoch in range(num_epochs):
                            print "epoch ::", epoch
                            model.train_epoch()
                        hit, total = accuracy.find_phrase_accuracy(model, samples, 5, False)
                        r5 = float(hit)/total
                        hit, total = accuracy.find_phrase_accuracy(model, samples, 1, False)
                        r1 = float(hit)/total
                        hit, total = accuracy.find_phrase_accuracy(model, training_samples, 1, False)
                        tr1 = float(hit)/total

                        with open("grid_results.txt","a") as testResultFile:
                            testResultFile.write('lr: ' + str(config.lr) +\
                                        '\tbatch_size ' + str(config.batch_size) +\
                                        '\thidden_size ' + str(config.hidden_size) +\
                                        '\tlayer1_size ' + str(config.layer1_size) +\
                                        '\tlayer2_size ' + str(config.layer2_size) +\
                                        '\tlayer3_size ' + str(config.layer2_size) +\
                                        '\tr5: '+ str(r5) +\
                                        '\tr1: '+ str(r1) +\
                                        '\ttr1: '+ str(tr1)+ "\n")                            

def test(repdir, config):
    rd = reader.Reader(open("data/hp.obo"), False)

    samples = accuracy.prepare_phrase_samples(rd, open("data/labeled_data"), True)
    training_samples = {}
    for hpid in rd.names:
        for s in rd.names[hpid]:
            training_samples[s]=[hpid]

    model = phrase_model.NCRModel(config, rd)
    model.load_params(repdir)

    hit, total = accuracy.find_phrase_accuracy(model, samples, 5, False)
    r5 = float(hit)/total
    print "R@5 Accuracy on test set ::", r5
    hit, total = accuracy.find_phrase_accuracy(model, samples, 1, False)
    r1 = float(hit)/total
    print "R@1 Accuracy on test set ::", r1
    hit, total = accuracy.find_phrase_accuracy(model, training_samples, 1, False)
    tr1 = float(hit)/total
    print "R@1 Accuracy on training set ::", tr1

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--repdir', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="checkpoints/")
    args = parser.parse_args()

    config = phraseConfig.Config

    #test(args.repdir, config) 
    #exit()

    #grid_search()
    model = new_train(config)
    model.save_params(args.repdir)

if __name__ == "__main__":
	main()

