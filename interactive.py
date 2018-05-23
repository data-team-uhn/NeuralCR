import argparse
import ncrmodel 
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--fasttext', help="address to the fasttext word vector file")
    parser.add_argument('--params', help="address to the directroy where the trained model parameters are stored")
    parser.add_argument('--threshold', type=float, help="the score threshold for concept recognition", default=0.8)
    args = parser.parse_args()
    
    print('Loading model...')
    model = ncrmodel.NCRModel.loadfromfile(args.params, args.fasttext)
    for line in sys.stdin:
        if line.startswith("->"):
            results = model.get_match(line[2:], 5)
            for r in results:
                print(r[0], model.ont.names[r[0]][0], r[1])
            print("")

        else:
            results = model.annotate_text(line, args.threshold)
            for res in results:
                print('\t'.join(map(str,res[:3]))+'\t'+model.ont.names[res[2]][0]+'\t'+str(res[3]))

if __name__ == "__main__":
    main()
