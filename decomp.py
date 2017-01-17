import fasttext_reader as reader
import numpy as np
from sklearn.decomposition import NMF
import os.path
import h5py

def get_anc_decomp(rd):
    anc_mask_decomp_file = 'data/anc_mask.h5'
    if True or not os.path.exists(anc_mask_decomp_file):
        x = rd.ancestry_mask
        #x = np.array ([[1,2,3],[1,3,3],[4,2,3],[1,2,3]])
        model = NMF(n_components=200, tol=.01)
        W = model.fit_transform(x);
        H = model.components_;
        print model.reconstruction_err_

        '''
        h5f = h5py.File(anc_mask_decomp_file, 'w')
        h5f.create_dataset('W', data=W)
        h5f.create_dataset('H', data=H)
        h5f.close()
        '''
        er = np.abs(x-np.matmul(W,H))
        print np.mean(er)
        print np.max(er)
        print len([xx for xx in er.flat if xx > 0.2])
        return W,H
    else:
        h5f = h5py.File(anc_mask_decomp_file, 'r')
        W = np.array(h5f['W'])
        H = np.array(h5f['H'])
        h5f.close()
        return W,H



def main():
    oboFile = open("data/hp.obo")
    rd = reader.Reader(oboFile) 
    w,h = get_anc_decomp(rd)
    print np.matmul(w,h)


if __name__ == "__main__":
	main()
