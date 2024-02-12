import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from lib import Autoencoder
from lib import FullyConnected
from config import path

autoencoder = Autoencoder()
autoencoder.built = True
autoencoder.load_weights(path + 'Autoencoder/weights/weights.hdf5')
fullyconnected = FullyConnected()
fullyconnected.built = True
fullyconnected.load_weights(path + 'FC/weights/weights.hdf5')

FullModel = tf.keras.Sequential([fullyconnected, autoencoder.decoder])


number = 3000

def create_heatmap_1(position, x):
    one_d_array = np.sum(x, axis=1)

    position = int(position)
    # array_split
    array_list = np.split(one_d_array, [position - 3, position + 3])

    # sums/scoring
    heatmap = []

    size0 = 0
    size2 = array_list[2].size
    for x in array_list[0]:
        heatmap.append(x * size0 ** 0.75 / array_list[0].size ** 0.75)
        size0 += 1

    for x in array_list[1]:
        heatmap.append(0)

    for x in array_list[2]:
        heatmap.append(x * size2 ** 0.75 / array_list[2].size ** 0.75)
        size2 -= 1

    return heatmap



def visualise(array):
    # encode image
    '''
    array[5] = number / 10000
    array = np.expand_dims(array, axis=0)
    temp = fullyconnected.predict(np.array(array))
    picture = autoencoder.decoder.predict(temp)[0]
    '''
    fig, ax1 = plt.subplots(1, 1)
    neg = ax1.imshow(array, cmap='Reds_r', interpolation='none')
    fig.colorbar(neg, ax=ax1, location='right', anchor=(0, 0.3), shrink=0.7)
    #ax1.imshow(array)
    #fig.savefig("pic_" + str(number) + ".png")
    fig.savefig("heatmap.png")

array = np.load(path + '3000_best_x.npy')
x = array[-1]

output_variable = x * np.array([150, 3, 800, 100, 100, 10_000, 3])
for value in output_variable:
    print(format(value, '.8f'))

array_of_ones = np.ones((256, 256))

heat = create_heatmap_1(256/2, array_of_ones)
heat = np.array(heat).reshape(256,1)
#heat = np.array(heat).transpose()
print(heat)
heat = np.repeat(heat, 256, axis=1)
visualise(heat)
