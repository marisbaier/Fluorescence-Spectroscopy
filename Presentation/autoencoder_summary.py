'''
Semesterproject Fluorescence Spectroscopy

Diese Datei soll die loss-history für den Autoencoder und finale inference plots darstellen.
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))[:-12])

from lib import Autoencoder
from config import path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


autoencoder = Autoencoder()
autoencoder.built = True
autoencoder.load_weights(path+'Autoencoder/weights/weights.hdf5')

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,6))

# Plot loss history
history = np.load(path+'history.npy', allow_pickle=True).item()

#fig.suptitle('Loss History of the Autoencoder')
fig.subplots_adjust(hspace=0.7)

ax1.plot(history['loss'], label='loss', color='black')
ax1.plot(history['val_loss'], label='validaiton loss', color='red')
ax1.set_xlabel('Samples seen [x50000]')
ax1.set_ylabel('Loss')
ax1.set_title('Loss History')
ax1.legend()
ax1.grid()

ax2.plot(history['loss'], label='loss', color='black')
ax2.plot(history['val_loss'], label='validation loss', color='red')
ax2.set_xlim(250, 300)
ax2.set_ylim(0, 0.03)
ax2.set_xlabel('Samples seen [x50000]')
ax2.set_ylabel('Loss')
ax2.set_title('Loss History (Zoomed)')
ax2.legend()
ax2.grid()

plt.savefig('Presentation/Autoencoder_Loss_History.png')
