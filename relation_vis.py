import argparse
import pickle 

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('input', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'")
    parser.add_argument('params', help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'")
    parser.add_argument('--top', type=int, help="The location where the checkpoints and the logfiles will be stored, default is \'checkpoints/\'", default=20)
    args = parser.parse_args()

    cor = pickle.load(open(args.input,'rb')) 
    ont = pickle.load(open(args.params+'/ont.pickle',"rb" )) 
    #print(len(cor))
    rel_final = [] 
    for r in cor:
        rel_final.append((r[0], r[1], len(cor[r])))

    rel_final.sort(key=lambda x:x[2], reverse=True)
    for r in rel_final[:args.top]:
        print(ont.names[r[0]][0]+'\t'+ont.names[r[1]][0]+'\t'+str(r[2]))
        #print (ont.names[r[0]][0], ont.names[r[1]][0], r[2], cor[(r[0],r[1])])

if __name__ == "__main__":
    main()
