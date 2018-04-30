import argparse
import ncrmodel 
import os

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--fasttext', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--params', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--input', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--output', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--threshold', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default=0.5)
    args = parser.parse_args()
    
    print('Loading model...')
    model = ncrmodel.NCRModel.loadfromfile(args.params, args.fasttext)
    print('Model loaded')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for filename in os.listdir(args.input):    
        print(filename)
        results = model.annotate_text(open(args.input+'/'+filename).read(), threshold = float(args.threshold))
        fp = open(args.output+'/'+filename,'w')
        for res in results:
            fp.write('\t'.join(map(str,res))+'\n')

if __name__ == "__main__":
    main()
