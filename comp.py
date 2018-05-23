from eval import normalize
import argparse
from onto import Ontology
import os

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('label_file', help="Path to the directory where the input text files are located")
    parser.add_argument('output_file1', help="Path to the directory where the output files will be stored")
    parser.add_argument('output_file2', help="Path to the directory where the output files will be stored")
    parser.add_argument('fileid', help="Path to the directory where the output files will be stored")
    parser.add_argument('--obofile', help="address to the ontology .obo file")
    parser.add_argument('--oboroot', help="the concept in the ontology to be used as root (only this concept and its descendants will be used)")
    args = parser.parse_args()

    ont = Ontology(args.obofile, args.oboroot)

    labels = normalize(ont, args.label_file+'/'+args.fileid)
    ncr = normalize(ont, args.output_file1+'/'+args.fileid, 2)
    biolark = normalize(ont, args.output_file2+'/'+args.fileid)

    sets = {"FALSE NEG -> labels & biolark - ncr: ":((labels & biolark)-ncr),
            "TRUE POS: -> labels & ncr - biolark: ":((labels & ncr)-biolark),
            "FALSE POS -> ncr - (labels | biolark): ":(ncr - (labels | biolark)),
            "TRUE NEG -> biolark - (labels | ncr): ":(biolark - (labels | ncr))
            }
    for s in sets:
        print(s) #,sets[s])
        for x in sets[s]:
            print (x,ont.names[x])
        print("")
    print("==============================\n")

if __name__ == "__main__":
	main()

