import argparse
import ncrmodel 
import os
import sys

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

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    file_list = os.listdir(args.input)
    for i,filename in enumerate(file_list):    
        sys.stdout.write("\rProgress:: %d%%" % (100*i//len(file_list)))
        sys.stdout.flush()
        results = model.annotate_text(open(args.input+'/'+filename).read(), threshold = float(args.threshold))
        fp = open(args.output+'/'+filename,'w')
        for res in results:
            fp.write('\t'.join(map(str,res))+'\n')
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()
