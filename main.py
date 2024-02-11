'''
Semesterproject Fluorescence-Spectroscopy
'''
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import optimize
from tensorflow import keras
from scipy.optimize import basinhopping

import assets.Autoencoder

'''
Funktionen und Zusammenhänge

Optimizer-Funktion  <---      
|	|		\			
|	V		|			
|    optimize(x)	| Score  	
|    	|		|	
|    	V		|
|    evaluate(input) ---
|	
|	
|--> visualise(array)

'''

#Visualisation-Code: safe pictures after finished optimisation
def visualise(array):
    # encode image
    array[5] = np.log(number + 1)
    array = np.expand_dims(array, axis=0)
    temp = model.predict(np.array(array))
    picture = autoencoder.decoder.predict(temp)[0]

    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(picture)
    fig.savefig("pic_"+str(number)+".png")

    
# constraints: definition of constraints for optimisation

def func(x):
    x = np.exp(x) - 1

    return np.array([-np.abs(x[1] - 2.25) + 0.75,
                     -np.abs(x[6] - 3.75) + 3.25,
                     -np.abs(x[0] - 110) + 40,
                     -np.abs(x[2] - 550) + 250,
                     -np.abs(x[3] - 50) + 50,
                     -np.abs(x[4] - 50) + 50])

cons = {'type': 'ineq',
        'fun': lambda x: func(x)}

#evalution function
def evaluate(input):
    index = 0
    max = 0.0
    position = 0

    # create one-dimensional array of row sums
    one_d_array = np.log(1 + (np.sum(input, axis=1)))
    '''
    #find mangan point
    for x in one_d_array:
        if x>=max:
            max = x
            position = index
            index += 1
        else:
            index += index+1
    '''
    max_thresh = 0.9 * input.max()

    ys_max = []
    for x in range(256):
        for y in range(256):
            if input[y, x] >= max_thresh:
                ys_max.append(y)
    position = np.mean(np.array(ys_max))
    position = int(position // 1)

    # array_split
    array_list = np.split(one_d_array, [position - 3, position + 3])

    # sums/scoring
    a, b, c = 0, 0, 0

    size0 = 0
    size2 = array_list[2].size
    for x in array_list[0]:
        a = a + x * size0**0.75 / array_list[0].size**0.75
        size0 += 1

    for x in array_list[2]:
        b = b + x * size2**0.75 / array_list[2].size**0.75
        size2 -= 1

    c = np.abs(128 - position) / 128 * 200

    if max_thresh > 5:
        score = a + b + c
    else:
        score = a + b + c + 1000

    return score

#initialise autoencoder and fully connected network
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
    ])
    return model

model = create_model()
model.load_weights(filepath='weights/weightsFC.hdf5')

autoencoder = assets.Autoencoder.Autoencoder()
autoencoder.built = True
autoencoder.load_weights(filepath='weights/weights.hdf5')

#optimize: create picture from trained network for given values (x),
#	   than score said picture with evaluate funktion, and return score to 
#	   optimizer

def optimize(x):
    # encode image
    x[5] = np.log(number+1)
    x = np.expand_dims(x, axis=0)
    temp = model.predict(np.array(x))
    picture = autoencoder.decoder.predict(temp)[0]

    # evaluate image
    score = evaluate(picture)

    return score


#defining starting variables for 3 unique cases
x3000 = np.log(1 + np.array([7.70011971e+01, 2.42572106e+00, 6.02109219e+02, 2.11285258e-01,
                             7.37786031e+01, 3.00000000e+03, 9.28045713e-01, 4.70011971e+01,
                             6.02109219e+02, 2.42572106e+00]))
x5000 = np.log(1 + np.array([1.42699906e+02, 2.44119493e+00, 7.50803173e+02, 6.88605452e+01,
                             6.86057883e+01, 5.00000000e+03, 9.72945235e-01, 1.12699906e+02,
                             7.50803173e+02, 2.44119493e+00]))
x10000 = np.log(1 + np.array([1.38591463e+02, 1.91580887e+00, 5.49097803e+02, 5.70393449e+00,
                              3.51447583e+01, 1.00000000e+04, 5.09269345e-01, 1.08591463e+02,
                              5.49097803e+02, 1.91580887e+00]))


#call optimizer for all cases, and safe most optimal picture
number = 3000
res = basinhopping(optimize, x3000, niter=100, minimizer_kwargs={'constraints': cons}, T=1.2)
visualise(np.array(res.x))
number = 5000
res = basinhopping(optimize, x5000, niter=100, minimizer_kwargs={'constraints': cons}, T=1.2)
visualise(np.array(res.x))
number = 10000
res = basinhopping(optimize, x10000, niter=100, minimizer_kwargs={'constraints': cons}, T=1.2)
visualise(np.array(res.x))


