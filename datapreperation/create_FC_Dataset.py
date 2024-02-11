'''
Semesterproject Fluorescence Spectroscopy
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))[:-15])

import numpy as np
import tensorflow as tf

from lib.Autoencoder import Autoencoder
from config import chunk_size, path


with tf.device('/device:GPU:2'):
    autoencoder = Autoencoder()
    autoencoder.built = True
    autoencoder.load_weights(path+'Autoencoder/weights/weights.hdf5')
    
    n=2
    labels = np.empty((n*chunk_size,7))
    latentspace = np.empty((n*chunk_size,32))
    for i in range(n):
        labels[i*chunk_size:(i+1)*chunk_size] = np.load(f'/home/tmp/baierluc/{i}_X.npy')[0:chunk_size,:7]
        Y = np.load(f'/home/tmp/baierluc/{i}_Y.npy')[:chunk_size,:,:,:]
        latentspace[i*chunk_size:(i+1)*chunk_size] = autoencoder.encoder.predict(Y)

    labels = np.divide(labels,[150,3,800,100,100,10_000,7])

    np.save(path+'FC/FC_dataset/labels.npy', labels)
    np.save(path+'FC/FC_dataset/latentspace.npy', latentspace)
