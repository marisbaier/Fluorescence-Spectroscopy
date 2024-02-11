'''
Semesterproject Fluorescence Spectroscopy
'''
import tensorflow as tf

from tensorflow.keras.models import Model


class FullyConnected(Model):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.Model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(7,)),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(32, activation='relu'),
                ])

    def call(self, x):
        return self.Model(x)

