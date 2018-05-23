import argparse
import pickle 
import ncrmodel 
import os
import sys


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
    parser.add_argument('--input', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--output', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--params', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    parser.add_argument('--sroot', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default="experiment")
    args = parser.parse_args()


    ont = pickle.load(open(args.params+'/ont.pickle',"rb" )) 
    print(ont.names[args.sroot])
    subset = set()
    for x in ont.names:
        if ont.concept2id[args.sroot] in ont.ancestrs[ont.concept2id[x]]:
            subset.add(x)
    
    wsize = 100
    maps = (int,int,str,float)
    file_list = os.listdir(args.input)
    cor={}
    for filename in file_list:    
        terms = [t.strip().split('\t') for t in open(args.input+'/'+filename).readlines()]
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
                    r = (terms[i], terms[j], filename)
                else:
                    key = (terms[j][2],terms[i][2])
                    r = (terms[j], terms[i], filename)

                if key not in cor:
                    cor[key] = []
                cor[key].append(r)
    
    pickle.dump(cor, open(args.output,'wb')) 
    print(len(cor))
    rel_final = [] 
    for r in cor:
        rel_final.append((r[0], r[1], len(cor[r])))

    rel_final.sort(key=lambda x:x[2], reverse=True)
    for r in rel_final[:20]:
        print (ont.names[r[0]][0], ont.names[r[1]][0], r[2], cor[(r[0],r[1])])

if __name__ == "__main__":
    main()
