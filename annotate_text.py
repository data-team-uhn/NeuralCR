import argparse
import ncrmodel 
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--fasttext', help="address to the fasttext word vector file")
    parser.add_argument('--params', help="address to the directroy where the trained model parameters are stored")
    parser.add_argument('--input', help="address to the directory where the input text files are located ")
    parser.add_argument('--output', help="adresss to the directory where the output files will be stored")
    parser.add_argument('--threshold', help="the score threshold for concept recognition", default=0.8)
    args = parser.parse_args()
    
    print('Loading model...')
    model = ncrmodel.NCRModel.loadfromfile(args.params, args.fasttext)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    file_list = os.listdir(args.input)
    for i,filename in enumerate(file_list):    
        sys.stdout.write("\rProgress:: %d%%" % (100*i//len(file_list)))
        sys.stdout.flush()
        text = open(args.input+'/'+filename).read()
        results = model.annotate_text(text, threshold = float(args.threshold))
        fp = open(args.output+'/'+filename,'w')
        for res in results:
            fp.write('\t'.join(map(str,res))+'\n')
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()
