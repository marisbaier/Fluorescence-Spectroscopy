'''
Semesterproject Fluorescence-Spectroscopy

This file plots the output from the final Network (FullyConnected -> autoencoder.decoder)
It also places sliders below so that the inputs can be changed dynamically.
'''
from lib.Autoencoder import Autoencoder
from lib.FullyConnected import FullyConnected
from config import path

import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
from matplotlib.widgets import Slider


autoencoder = Autoencoder()
autoencoder.built = True
autoencoder.load_weights(path+'Autoencoder/weights/weights.hdf5')
fullyconnected = FullyConnected()
fullyconnected.built = True
fullyconnected.load_weights(path+'FC/weights/weights.hdf5')

FullModel = tf.keras.Sequential([fullyconnected, autoencoder.decoder])

# Create a figure and a subplot for the output image
fig, ax = plt.subplots(figsize=(19,10))
plt.subplots_adjust(bottom=0.5)  # leave some space for the sliders

# Assuming you have a function to get the model output
def get_model_output(input1, input2, input3, input4, input5, input6, input7):
    # Replace this with your actual model
    input = np.array([input1, input2, input3, input4, input5, input6, input7]) / np.array([150,3,800,100,100,10_000,7])
    return FullModel.predict(input.reshape(1,7)).reshape(256,256)

# Initial plot
output = get_model_output(70, 1.5, 300, 0, 0, 3000, 0.5)
im = ax.imshow(output, vmin=0, vmax=np.log(1024))

# Function to update the plot
def update(val):
    output = get_model_output(slider1.val, slider2.val, slider3.val, slider4.val, slider5.val, slider6.val, slider7.val)
    im.set_data(output)
    fig.canvas.draw_idle()


#slider_range = [(70,150,1),(1.5,3.0,0.1),(300,800,10),(0.0,100.0,1),(0.0,100.0,1),(3000,10_000,1000),(0.5,7.0,0.1)]
# Create sliders
slider1 = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'Sagittal entrance arm length [70mm - 150mm]', 70, 150, valinit=70)
slider2 = Slider(plt.axes([0.25, 0.14, 0.65, 0.03]), 'Design angle alpha [1.5 - 3.0]', 1.5, 3.0, valinit=1.5)
slider3 = Slider(plt.axes([0.25, 0.18, 0.65, 0.03]), 'Sagittal exit arm length [300mm - 800mm]', 300, 800, valinit=300)
slider4 = Slider(plt.axes([0.25, 0.22, 0.65, 0.03]), 'Slope Error Sagittal (vertical axis) [0.0 - 100.0]', 0, 100, valinit=0)
slider5 = Slider(plt.axes([0.25, 0.26, 0.65, 0.03]), 'Slope Error Meridional (horizontal axis) [0.0 - 100.0]', 0, 100, valinit=0)
slider6 = Slider(plt.axes([0.25, 0.30, 0.65, 0.03]), 'Long Radius R [3000mm, 5000mm, 10000mm]', 3000, 10_000, valinit=3000)
slider7 = Slider(plt.axes([0.25, 0.34, 0.65, 0.03]), 'Design Angle beta [0.5 - 7.0] (eigentlich [0.5 - 3.0])', 0.5, 3.0, valinit=0.5)

# Update the plot when a slider is moved
slider1.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)
slider4.on_changed(update)
slider5.on_changed(update)
slider6.on_changed(update)
slider7.on_changed(update)

plt.show()
""" plt.savefig('FC.png')

slider1.val = 150
update(150)
plt.savefig('FC2.png')

slider3.val = 800
update(150)
plt.savefig('FC3.png')

slider4.val = 50
update(150)
plt.savefig('FC4.png')

slider5.val = 50
update(150)
plt.savefig('FC5.png')

slider6.val = 5_000
update(150)
plt.savefig('FC6.png')

slider7.val = 3.5
update(150)
plt.savefig('FC7.png') """
