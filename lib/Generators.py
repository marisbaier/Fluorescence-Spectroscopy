'''
Semesterproject Fluorescence Spectroscopy
'''
import h5py
import numpy as np

from config import path, batch_size
from lib import Autoencoder


class autoencoder_generator:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def __call__(self):
        chunk_index = np.random.randint(0, 38)
        Y = np.load(f'{self.file_path}{chunk_index}_Y.npy')
        print('\nloaded chunk: ',chunk_index)
        num_batches = len(Y)//batch_size
        for batch in np.split(Y[:batch_size*num_batches], num_batches):
            yield (batch, batch)

class FullyConnectedGenerator:
    def __init__(self, file_path):
        self.data = h5py.File(file_path, 'r')
        self.autoencoder = Autoencoder()
        self.autoencoder.built = True
        self.autoencoder.load_weights(path+'/weights/weights.hdf5')

    def __call__(self):
        """ for chunk_index in range(38):
             """
        for key in self.data.keys():
            obj = self.data[key]
            X = np.log(np.array(obj['X'][()]))
            Y = np.expand_dims(np.log(1+np.array(obj['Y'][()]).reshape(256,256,1)), axis=0)
            predic = self.autoencoder.encoder.predict(Y)
            yield (X, predic.reshape(32,))
