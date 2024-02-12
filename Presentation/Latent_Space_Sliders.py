'''
Semesterproject Fluorescence-Spectroscopy

This file plots the output from the final Network (FullyConnected -> autoencoder.decoder)
It also places sliders below so that the inputs can be changed dynamically.
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))[:-12])

from lib import Autoencoder
from lib import FullyConnected
from config import path
from scoring import evaluate

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button


autoencoder = Autoencoder()
autoencoder.built = True
autoencoder.load_weights(path+'Autoencoder/weights/weights.hdf5')
fullyconnected = FullyConnected()
fullyconnected.built = True
fullyconnected.load_weights(path+'FC/weights/weights.hdf5')

FullModel = tf.keras.Sequential([fullyconnected, autoencoder.decoder])

# Create a figure and a subplot for the output image
fig, (ax, ax2) = plt.subplots(1,2,figsize=(19,10))
plt.subplots_adjust(bottom=0.5)  # leave some space for the sliders

# Assuming you have a function to get the model output
def get_model_output(input):
    # Replace this with your actual model
    return autoencoder.decoder.predict(input.reshape(1,32)).reshape(256,256)

# Initial plot
output = get_model_output(np.ones(32))
im = ax.imshow(output, vmin=0, vmax=np.log(1024))

# Function to update the plot
def update(val=None):
    output = get_model_output(np.array([slider.val for slider in sliders]))
    im.set_data(output)
    fig.canvas.draw_idle()

# Create sliders
""" slider1 = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'Sagittal entrance arm length [70mm - 150mm]', 70, 150, valinit=150)
slider2 = Slider(plt.axes([0.25, 0.14, 0.65, 0.03]), 'Design angle alpha [1.5 - 3.0]', 1.5, 3.0, valinit=3)
slider3 = Slider(plt.axes([0.25, 0.18, 0.65, 0.03]), 'Sagittal exit arm length [300mm - 800mm]', 300, 800, valinit=507)
slider4 = Slider(plt.axes([0.25, 0.22, 0.65, 0.03]), 'Slope Error Sagittal (vertical axis) [0.0 - 100.0]', 0, 100, valinit=100)
slider5 = Slider(plt.axes([0.25, 0.26, 0.65, 0.03]), 'Slope Error Meridional (horizontal axis) [0.0 - 100.0]', 0, 100, valinit=0)
slider6 = Slider(plt.axes([0.25, 0.30, 0.65, 0.03]), 'Long Radius R [3000mm, 5000mm, 10000mm]', 3000, 10_000, valinit=10000)
slider7 = Slider(plt.axes([0.25, 0.34, 0.65, 0.03]), 'Design Angle beta [0.5 - 7.0] (eigentlich [0.5 - 3.0])', 0.5, 3.0, valinit=2.6) """

sliders = [Slider(plt.axes([0.25+(i>15)*0.3, 0.03*i-(i>15)*16*0.03, 0.3, 0.02]), f'latent space {i}', -20, 20, valinit=1) for i in range(32)]

# Update the plot when a slider is moved
for slider in sliders:
    slider.on_changed(update)

#btn = Button(plt.axes([0.8, 0.01, 0.1, 0.04]), 'Optimize')
#txt = plt.text(-3.89,15,'Current Score: 0.000')

""" def optimize(event):
    x = np.load(path+'scoring/3000_best_x.npy')
    scoring = np.load(path+'scoring/3000_scoring.npy')
    ax2.set_xlim(0,len(scoring))
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Tested Score')
    steps=5
    for current,x in enumerate(x[::steps]):
        ax2.clear()
        ax2.plot(scoring[:steps*current])
        x *= np.array([150,3,800,100,100,10_000,3])
        for i in range(7):
            eval(f'slider{i+1}.set_val({x[i]})')
        fig.canvas.draw_idle()
        plt.pause(0.0001)

btn.on_clicked(optimize) """

plt.show()
