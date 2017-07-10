import matplotlib.pyplot as plt
from tsne import bh_sne
import h5py
import numpy as np
import fasttext_reader as reader

def main():

    N=20

    h5f = h5py.File('embeddings.h5', 'r')
    W = np.array(h5f['W'])
    aggW = np.array(h5f['aggW'])
    y = np.array(h5f['y'])
    y_sum = np.minimum(N,np.array(h5f['y_sum']))
    dat = np.array(h5f['taggW'])

    h5f.close()
    #dat = bh_sne(aggW.astype(np.float))
    import matplotlib.pyplot as plt

    
    print y
    print y_sum
    cmap = plt.cm.get_cmap('jet')
    #cmap = cmap.from_list('cosum', cmap(np.linspace(0, 1, int(np.max(y))), int(np.max(y))))
    cmap = cmap.from_list('cosum', cmap(np.linspace(0, 1, np.max(y))), np.max(y))

    fig = plt.figure(figsize=(18, 18))
    plt.scatter(dat[:,0], dat[:,1], c=y, cmap=cmap)
    ct = 0
    for edge in open('graph.txt','r').readlines():
        q,p = map(int,edge.strip().split())
        plt.plot([dat[q,0],dat[p,0]], [dat[q,1],dat[p,1]], 'black', alpha=0.2)
        ct += 1
#    plt.colorbar(sc, cmap=cmap, ticks=range(int(np.max(y))))
    plt.savefig("top_heading.png",bbox_inches='tight')
    plt.close()
    #plt.show()


    cmap = plt.cm.get_cmap('jet')
    cmap = cmap.from_list('cosum', cmap(np.linspace(0, 1, N)), N)

    fig = plt.figure(figsize=(18, 18))
    sc = plt.scatter(dat[:,0], dat[:,1], c=y_sum, cmap=cmap)

#    plt.colorbar(cmap=cmap, ticks=range(N))
    plt.savefig("depth.png",bbox_inches='tight')
    #plt.show()

if __name__ == "__main__":
	main()
