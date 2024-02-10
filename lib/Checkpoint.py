'''
Semesterproject Fluorescence Spectroscopy
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, samplepath="/home/tmp/baierluc/1_Y.npy", save_freq=20):
        super(Checkpoint, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.loss_filepath = self.filepath + 'loss.npy'
        self.val_loss_filepath = self.filepath + 'val_loss.npy'
        self.weights_filepath = self.filepath + 'weights/weights.hdf5'
        self.loss = np.load(self.loss_filepath) if os.path.exists(self.loss_filepath) else []
        self.val_loss = np.load(self.val_loss_filepath) if os.path.exists(self.val_loss_filepath) else []
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.weights_filepath,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_freq=self.save_freq,
            save_best_only=True,
            verbose=1)
        
        self.samples = np.load(samplepath)[5:9]
        self.fig = plt.figure(figsize=(10,10), dpi=500)
        self.ax1 = plt.subplot(3,1,1)
        self.ax1.set_yscale('log')
        self.ax2 = plt.subplot(3,4,5)
        self.ax3 = plt.subplot(3,4,6)
        self.ax4 = plt.subplot(3,4,7)
        self.ax5 = plt.subplot(3,4,8)
        self.ax6 = plt.subplot(3,4,9)
        self.ax7 = plt.subplot(3,4,10)
        self.ax8 = plt.subplot(3,4,11)
        self.ax9 = plt.subplot(3,4,12)
        self.epochs_done = 0

    """ def on_batch_end(self, batch, logs=None):
        self.loss = np.append(self.loss, logs['loss']) """
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_done += 1
        self.model.save_weights(self.filepath+f'weights/weights_epoch{self.epochs_done}.hdf5')
        """ 
        self.val_loss = np.append(self.val_loss, logs['val_loss'])
        self.loss = np.append(self.loss, logs['loss'])
        self.ax1.plot(self.loss, color='black', label='loss')
        self.ax1.plot(self.val_loss, color='red', label='val_loss')
        #self.ax1.legend()

        predictions = self.model.predict(self.samples)
        self.ax2.imshow(predictions[0].reshape(256,256), vmin=0, vmax=np.log(1024))
        self.ax3.imshow(predictions[1].reshape(256,256), vmin=0, vmax=np.log(1024))
        self.ax4.imshow(predictions[2].reshape(256,256), vmin=0, vmax=np.log(1024))
        self.ax5.imshow(predictions[3].reshape(256,256), vmin=0, vmax=np.log(1024))
        self.ax6.imshow(self.samples[0].reshape(256,256), vmin=0, vmax=np.log(1024))
        self.ax7.imshow(self.samples[1].reshape(256,256), vmin=0, vmax=np.log(1024))
        self.ax8.imshow(self.samples[2].reshape(256,256), vmin=0, vmax=np.log(1024))
        self.ax9.imshow(self.samples[3].reshape(256,256), vmin=0, vmax=np.log(1024))
        self.fig.savefig(self.filepath + '/loss.png') """
        np.save(self.loss_filepath, np.array(self.loss))
        np.save(self.val_loss_filepath, np.array(self.val_loss))

    def on_train_end(self, logs=None):
        np.save(self.loss_filepath, np.array(self.loss))
        np.save(self.val_loss_filepath, np.array(self.val_loss))
