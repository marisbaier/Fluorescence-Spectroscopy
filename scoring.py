'''
Semesterproject Fluorescence-Spectroscopy
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import basinhopping

from lib import Autoencoder
from lib import FullyConnected
from config import path


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
if __name__ == "__main__":
    # Visualisation-Code: safe pictures after finished optimisation
    def visualise(array):
        # encode image
        array[5] = number / 10000
        array = np.expand_dims(array, axis=0)
        temp = fullyconnected.predict(np.array(array))
        picture = autoencoder.decoder.predict(temp)[0]

        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(picture)
        fig.savefig(path+"scoring/pic_" + str(number) + ".png")


    # constraints: definition of constraints for optimisation

    def func(x):
        y = x * np.array([150, 3, 800, 100, 100, 10_000, 3])
        return np.array([-np.abs(y[0] - 110) + 40,
                        -np.abs(y[1] - 2.25) + 0.75,
                        -np.abs(y[2] - 550) + 250,
                        -np.abs(y[3] - 50) + 50,
                        -np.abs(y[4] - 50) + 50,
                        1,
                        -np.abs(y[6] - 3.75) + 3.25])


    cons = {'type': 'ineq',
            'fun': lambda x: func(x)}

    best_x = []
    score_array = []
    best_score = 100000


    def safe_score(score, x):
        global best_score
        score_array.append(score)
        #if score < best_score and (func(x) >= 0).all():
        best_x.append(x)
        #best_score = score


    # evalution function
    def evaluate(input):
        index = 0
        max = 0.0
        position = 0

        # create one-dimensional array of row sums
        one_d_array = np.sum(input, axis=1)
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
        max_thresh = 0.98 * input.max()

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
            a = a + x * size0 ** 0.5 / array_list[0].size ** 0.5
            size0 += 1

        for x in array_list[2]:
            b = b + x * size2 ** 0.5 / array_list[2].size ** 0.5
            size2 -= 1

        c = np.abs(128 - position) / 128 * 200

        if max_thresh > 5:
            score = a + b + c
        else:
            score = a + b + c + 1000

        return score


    # initialise autoencoder and fully connected network
    autoencoder = Autoencoder()
    autoencoder.built = True
    autoencoder.load_weights(path + 'Autoencoder/weights/weights.hdf5')
    fullyconnected = FullyConnected()
    fullyconnected.built = True
    fullyconnected.load_weights(path + 'FC/weights/weights.hdf5')

    FullModel = tf.keras.Sequential([fullyconnected, autoencoder.decoder])


    # optimize: create picture from trained network for given values (x),
    #	   than score said picture with evaluate funktion, and return score to
    #	   optimizer

    def optimizer(x):
        # encode image


        x[5] = number / 10_000
        #print(np.shape(x))
        #print(x)
        picture = FullModel.predict(np.array(x).reshape(1, 7))[0]

        # evaluate image
        score = evaluate(picture)

        safe_score(score, x)

        return score


    # defining starting variables for 3 unique cases
    x3000 = np.array([7.70011971e+01, 2.42572106e+00, 6.02109219e+02, 2.11285258e-01,
                    7.37786031e+01, 3.00000000e+03, 9.28045713e-01]) / np.array([150, 3, 800, 100, 100, 10_000, 3])
    x5000 = np.array([1.42699906e+02, 2.44119493e+00, 7.50803173e+02, 6.88605452e+01,
                    6.86057883e+01, 5.00000000e+03, 9.72945235e-01]) / np.array([150, 3, 800, 100, 100, 10_000, 3])
    x10000 = np.array([1.38591463e+02, 1.91580887e+00, 5.49097803e+02, 5.70393449e+00,
                    3.51447583e+01, 1.00000000e+04, 5.09269345e-01]) / np.array([150, 3, 800, 100, 100, 10_000, 3])

    # call optimizer for all cases, and safe most optimal picture
    number = 3000
    res1 = basinhopping(optimizer, x3000, niter=6, minimizer_kwargs={'constraints': cons, 'method':"COBYLA"}, T=1.2)
    visualise(np.array(res1.x))


    np.save(path + 'scoring/3000_scoring.npy', np.array(score_array))
    np.save(path + 'scoring/3000_best_x.npy', np.array(best_x))
    best_x = []
    score_array = []
    best_score = 100000

    number = 5000
    res2 = basinhopping(optimizer, x5000, niter=6, minimizer_kwargs={'constraints': cons, 'method':"COBYLA"}, T=1.2)
    visualise(np.array(res2.x))


    np.save(path + 'scoring/5000_scoring.npy', np.array(score_array))
    np.save(path + 'scoring/5000_best_x.npy', np.array(best_x))
    best_x = []
    score_array = []
    best_score = 100000

    number = 10000
    res3 = basinhopping(optimizer, x10000, niter=6, minimizer_kwargs={'constraints': cons, 'method':"COBYLA"}, T=1.2)
    visualise(np.array(res3.x))
    output_variable = res3.x * np.array([150, 3, 800, 100, 100, 10_000, 3])
    for value in output_variable:
        print(format(value, '.8f'))
    np.save(path + 'scoring/10000_scoring.npy', np.array(score_array))
    np.save(path + 'scoring/10000_best_x.npy', np.array(best_x))

    print(res1)
    print(res2)
    print(res3)
