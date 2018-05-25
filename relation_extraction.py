import argparse
import pickle 
import ncrmodel 
import os
import sys
import csv

class CSVInputStream:
    def __init__(self, input_csv_file, max_rows=-1):
        self.input_csv_file = input_csv_file
        if max_rows==-1:
            with open(self.input_csv_file) as fp:
                self.length = sum(1 for row in fp)
        else:
            self.length=max_rows

    def __len__(self):
        return self.length

    def __iter__(self):                           
        self.csvfile = open(self.input_csv_file, 'r') 
        self.reader = csv.reader(self.csvfile, delimiter=';')
        self.csv_iter = iter(self.reader)
        self.ct=0
        return self

    def __next__(self):                           
        self.ct+=1
        if self.ct>self.length:
            self.csvfile.close()
            raise StopIteration

        try:
            row = next(self.csv_iter)
        except StopIteration:
            self.csvfile.close()
            raise StopIteration
        terms = [[x.strip() for x in term.replace('(','').replace(')','').replace("'",'').split(',')] for term in row[1:]]
        return row[0], terms

def atomic(s, modela, modelb, ta,tb):
    res_a = modela.annotate_text(s, threshold=ta)
    ans = set()
    if len(res_a)==0:
        return ans
    res_b = modelb.annotate_text(s, threshold=tb)
    if len(res_b)==0:
        return ans

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    #parser.add_argument('--fasttext', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
#    parser.add_argument('--modela', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
#    parser.add_argument('--modelb', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    #parser.add_argument('--inputa', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--input', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'")
    parser.add_argument('--output', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'")
    parser.add_argument('--params', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'")
    parser.add_argument('--sroot', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'")
    args = parser.parse_args()


    ont = pickle.load(open(args.params+'/ont.pickle',"rb" )) 
    subset = set()
    for x in ont.names:
        if ont.concept2id[args.sroot] in ont.ancestrs[ont.concept2id[x]]:
            subset.add(x)
    input_stream = CSVInputStream(args.input)
    
    wsize = 100
    maps = (int,int,str,float)
    #file_list = os.listdir(args.input)
    cor={}
    #for filename in file_list:    
    for key,terms in input_stream:
        #terms = [t.strip().split('\t') for t in open(args.input+'/'+filename).readlines()]
        for term in terms:
            for i in range(len(maps)):
                term[i]=maps[i](term[i])
        terms.sort(key=lambda x:x[0])
        for i in range(len(terms)):
            for j in range(i+1, len(terms)):
                if terms[j][0] - terms[i][1] > wsize:
                    break
                if (terms[i][2] in subset) == (terms[j][2] in subset):
                    continue

                if terms[i][2] not in subset:
                    key = (terms[i][2],terms[j][2])
                    r = (terms[i], terms[j], key)
                else:
                    key = (terms[j][2],terms[i][2])
                    r = (terms[j], terms[i], key)

                if key not in cor:
                    cor[key] = []
                cor[key].append(r)
    
    pickle.dump(cor, open(args.output,'wb')) 

if __name__ == "__main__":
    main()
