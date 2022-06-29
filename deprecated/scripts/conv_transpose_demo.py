# Demo of transposed convolution in 2d
# Based on code from https://machinelearningmastery.com/generative_adversarial_networks/

import superimport

import tensorflow as tf
from numpy import asarray
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose

# 2d input
X = asarray([[1, 2],
			 [3, 4]])
print(X)

# reshape input data into one sample a sample with a channel
X = X.reshape((1, 2, 2, 1))

# define model: we use 1 kernel of size  1x1 and use stride 2  to upsample by 2x
model = Sequential()
model.add(Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=(2, 2, 1)))

# define weights so that they do nothing: weight=1, bias=0
weights = [asarray([[[[1]]]]), asarray([0])]
model.set_weights(weights)

yhat = model.predict(X)
# reshape output to remove channel to make printing easier
yhat = yhat.reshape((4, 4))
print(yhat)

'''
[[1. 0. 2. 0.]
 [0. 0. 0. 0.]
 [3. 0. 4. 0.]
 [0. 0. 0. 0.]]
'''

# example of using padding=same to ensure that the output size is exactly doubled
# even though kernel has size 3x3
model = Sequential()
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding='same', input_shape=(2, 2, 1)))
yhat = model.predict(X)
print(yhat.shape) # (1,4,4,1)
