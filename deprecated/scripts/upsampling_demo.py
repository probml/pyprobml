# Illustrate upsampling in 2d
# Code from Jason Brownlee
# https://machinelearningmastery.com/generative_adversarial_networks/


import superimport

import tensorflow as tf
from tensorflow import keras

from numpy import asarray

#from keras.models import Sequential
from tensorflow.keras.models import Sequential

#from keras.layers import UpSampling2D
from tensorflow.keras.layers import UpSampling2D

X = asarray([[1, 2],
			 [3, 4]])

X = asarray([[1, 2, 3],
			 [4, 5, 6],
             [7,8,9]])
print(X)
nr = X.shape[0]
nc = X.shape[1]
 
# reshape input data into one sample a sample with a channel
X = X.reshape((1, nr, nc, 1))

model = Sequential()
model.add(UpSampling2D(input_shape=(nr, nc, 1))) # nearest neighbor

yhat = model.predict(X)
yhat = yhat.reshape((2*nr, 2*nc))
print(yhat)

model = Sequential()
model.add(UpSampling2D(input_shape=(nc, nc, 1), interpolation='bilinear'))

yhat = model.predict(X)
yhat = yhat.reshape((2*nr, 2*nc))
print(yhat)