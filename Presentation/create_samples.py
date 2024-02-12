'''
Semesterproject Fluorescence-Spectroscopy

This file should create (log-scaled) sample images in Presentation/samples
'''
import numpy as np
import matplotlib.pyplot as plt


Y = np.load('/home/tmp/baierluc/1_Y.npy')

for i in range(20):
    plt.imshow(Y[i])
    plt.savefig(f'samples/{i}.png')
