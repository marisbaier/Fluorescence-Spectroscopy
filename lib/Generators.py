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
