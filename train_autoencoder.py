'''
Semesterproject Fluorescence Spectroscopy
'''
import numpy as np
import tensorflow as tf

from config import path, batch_size, epochs, optimizer, loss

from lib.Autoencoder import Autoencoder
from lib.Generators import autoencoder_generator
from lib.Checkpoint import Checkpoint


if __name__ == '__main__':
    print('available gpus: ', tf.config.list_physical_devices('GPU'))
    
    def dataset(path_to_data):
        return tf.data.Dataset.from_generator(
            autoencoder_generator(path_to_data),
            output_signature = (
                tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
            )
        )

    with tf.device('/device:GPU:2'): 
        autoencoder = Autoencoder()
        #autoencoder.built=True
        autoencoder.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        #autoencoder.load_weights('weights(1).hdf5')

        data = dataset('/home/tmp/baierluc/') # Auf gruenau9 liegen die entpackten Daten in Form von numpy arrays
        val = np.load('/home/tmp/baierluc/1_Y.npy')
        
        history = autoencoder.fit(
            data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                Checkpoint(path+'Autoencoder/', save_freq=1000).model_checkpoint_callback,
                Checkpoint(path+'Autoencoder/')
                ],
            validation_data=(val, val),
            #verbose=2,
        )

        np.save(path+'history.npy', history.history)
