'''
Semesterproject Fluorescence Spectroscopy
'''
import numpy as np
import tensorflow as tf

from lib.Checkpoint import Checkpoint
from lib.FullyConnected import FullyConnected
from train_autoencoder import path, loss, optimizer, batch_size, epochs


if __name__ == '__main__':
    epochs = 300
    print('available gpus: ', tf.config.list_physical_devices('GPU'))

    with tf.device('/device:GPU:2'):
        fullyconnected = FullyConnected()
        fullyconnected.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        fullyconnected.built = True
        fullyconnected.summary()

        # train data
        labels = np.load(path+'FC/FC_dataset/labels.npy')
        latentspace = np.load(path+'FC/FC_dataset/latentspace.npy')

        history = fullyconnected.fit(
            labels, 
            latentspace,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=Checkpoint(path+'FC/', samplepath=path+'FC/FC_dataset/labels.npy', save_freq=20),
            validation_data = (labels[:1000], latentspace[:1000])
        )

        np.save(path+'FC_history.npy', history.history)
