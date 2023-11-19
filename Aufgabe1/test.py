import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

print('tensorflow version: ',tf.__version__)
print('available gpus: ', tf.config.list_physical_devices('GPU')) # gpus?