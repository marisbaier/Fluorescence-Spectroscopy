'''
Semesterproject Fluorescence-Spectroscopy
'''

import h5py
import numpy as np
from config import chunk_size

if __name__ == '__main__':
    data = h5py.File('/dev/shm/rzp-1_sphere1mm_train_2million.h5', 'r')
    
    for i in range(2*10**6//chunk_size):
        print('Chunks saved: ', i)
        X = np.empty((chunk_size, 10,), dtype=np.float32)
        Y = np.empty((chunk_size, 256, 256, 1), dtype=np.float32)

        for key in range(i*chunk_size, (i+1)*chunk_size):
            print('datapoints visited :', key, end='\r', flush=True)
            obj = data[str(key)]
            X[key % chunk_size] = np.array(obj['X'][()])
            Y[key % chunk_size] = np.array(obj['Y'][()]).reshape((256,256,1))

        np.save(f'/home/tmp/baierluc/{i}_{"X"}.npy', X)
        np.save(f'/home/tmp/baierluc/{i}_{"Y"}.npy', np.log(1+Y))