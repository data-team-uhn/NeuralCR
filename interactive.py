import argparse
import ncrmodel 
import os
import sys

def main():
  parser = argparse.ArgumentParser(description='Hello!')
  parser.add_argument('--fasttext', help="address to the fasttext word vector file")
  parser.add_argument('--params', help="address to the directroy where the trained model parameters are stored")
  parser.add_argument('--threshold', type=float, help="the score threshold for concept recognition", default=0.75)
  args = parser.parse_args()
  
  model = ncrmodel.NCR.loadfromfile(args.params, args.fasttext)
  print("Enter a sentence to be annotated. For entity linking start the query with \'>\' followed by the name of the entity. Enter query:")
  for line in sys.stdin:
    if line.startswith(">"):
      results = model.get_match(line[1:], 5)
      for r in results:
        print(r[0], (model.ont.names[r[0]][0] if r[0]!='None' else r[0]), r[1])
      print("")

    else:
      results = model.annotate_text(line, args.threshold)
      for res in results:
        print('\t'.join(map(str,res[:3]))+'\t'+model.ont.names[res[2]][0]+'\t'+str(res[3]))
    print("Enter a sentence to be annotated. For entity linking start the query with \'>\' followed by the name of the entity. Enter query:")

if __name__ == "__main__":
  main()
