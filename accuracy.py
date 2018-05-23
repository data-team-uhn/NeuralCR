import argparse
import ncrmodel 

def prepare_phrase_samples(ont, samplesFile, filter_flag=False):
    samples = {}
    for line in open(samplesFile):
        tokens = line.strip().split("\t")
        if len(tokens)>2:
            continue
        real_hp_id = ont.real_id[tokens[1].strip().replace("_",":")]
        if real_hp_id not in ont.concepts:
                continue
        samples[tokens[0].strip()] = real_hp_id
    if filter_flag:
        copy_samples = dict(samples)
        for x in copy_samples:
            normed = x.lower()
            if normed in ont.name2conceptid:
                del samples[x]

    return samples


#################### Accuracy function ####################

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--fasttext', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--params', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--input', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--topk', type=int, help="", default=1)
    args = parser.parse_args()
    
    model = ncrmodel.NCRModel.loadfromfile(args.params, args.fasttext)

    samples = prepare_phrase_samples(model.ont, args.input, True)

    res = model.get_match(list(samples.keys()), args.topk)
    missed = [x for i,x in enumerate(samples) if samples[x] not in [r[0] for r in res[i]]]
    print(len(missed), len(samples), (len(samples)-len(missed))/len(samples))


if __name__ == "__main__":
    main()

