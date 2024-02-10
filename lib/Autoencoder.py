'''
Semesterproject Fluorescence Spectroscopy
'''
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers


class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential(
      [
        layers.Input(shape=(256, 256)),
        layers.Reshape((256, 256, 1)),
        layers.Conv2D(16, (9, 9), activation='relu', padding='same', strides=4),    # 256*256*1 -> 64x64x16
        layers.Conv2D(32, (5,5), activation='relu', padding='same', strides=2),     # 64x64x16 -> 32x32x32
        layers.MaxPooling2D(pool_size=(2, 2),strides=None,padding='same'),          # 32x32x32 -> 16x16x32
        layers.Normalization(),

        layers.Conv2D(48, (3,3), activation='relu', padding='same', strides=1),     # 16x16x32 -> 16x16x48
        layers.Conv2D(48, (3,3), activation='relu', padding='same', strides=1),     # 16x16x48 -> 16x16x48
        layers.MaxPooling2D(pool_size=(2, 2),strides=None,padding='same'),          # 16x16x48 -> 8x8x48
        layers.Normalization(),

        layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=1),     # 8x8x48 -> 8x8x64
        layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=1),     # 8x8x64 -> 8x8x64
        layers.MaxPooling2D(pool_size=(2, 2),strides=None,padding='same'),          # 8x8x64 -> 4x4x64
        layers.Normalization(),

        layers.Flatten(),                                                           # 4x4x64 -> 1024
        layers.Dense(128, activation='relu'),                                       # 1024 -> 128
        layers.Dense(32, activation='relu'),                                        # 128 -> 32
      ]
    )

    self.decoder = tf.keras.Sequential(
      [
        layers.Input(shape=(32,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Reshape((4,4,64)),

        layers.Normalization(),
        layers.UpSampling2D(size=(2, 2)),                                                   # 4x4x64 -> 8x8x64
        layers.Conv2DTranspose(64, (3,3), activation='relu', padding='same', strides=1),    # 8x8x64 -> 8x8x64 
        layers.Conv2DTranspose(48, (3,3), activation='relu', padding='same', strides=1),    # 8x8x64 -> 8x8x48

        layers.Normalization(),
        layers.UpSampling2D(size=(2, 2)),                                                   # 8x8x48 -> 16x16x48         
        layers.Conv2DTranspose(48, (3,3), activation='relu', padding='same', strides=1),    # 16x16x48 -> 16x16x48
        layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same', strides=1),    # 16x16x48 -> 16x16x32

        layers.Normalization(),
        layers.UpSampling2D(size=(2, 2)),                                                   # 16x16x32 -> 32x32x32
        layers.Conv2DTranspose(16, (5,5), activation='relu', padding='same', strides=2),    # 32x32x32 -> 64x64x16
        layers.Conv2DTranspose(1, (9,9), activation='relu', padding='same', strides=4),     # 64x64x16 -> 256x256x1
      ]
    )

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
