import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

print('tensorflow version: ',tf.__version__) # need <2.11 on win
print('available gpus: ', tf.config.list_physical_devices('GPU')) # gpu?

'''
Data preparation
'''

data = pd.read_csv('XFEL_KW0_Results_2.csv', names=None)

# find constant columns, else normalization wouldn't work
data = data.loc[:, (data != data.iloc[0]).any()] 

# label input 0-6, output 0-4 just for clarity
num_inputs = 7
num_outputs = 5
data.columns = np.concatenate(
    (["Input %s" % i for i in range(num_inputs)], ["Output %s" % i for i in range(num_outputs)])
)

