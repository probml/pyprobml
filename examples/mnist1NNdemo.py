#Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance

from scipy import sparse
import numpy as np

import struct
import os
import time

from examples import get_mnist
#(x_train, y_train, x_test, y_test) = get_mnist.get_mnist()

(TrainIms,TrainLabels,TestIms,TestLabels) = get_mnist.get_mnist()
print(TrainIms.shape,TrainIms.dtype)
print(TrainLabels.shape,TrainLabels.dtype)
print(TestIms.shape)
print(TestLabels.shape)

#Flattens images into sparse vectors. So we go from 3D to 2D image datasets.
def Flatten(Ims):
    return(sparse.csr_matrix(Ims.reshape(Ims.shape[0],-1)))

TrainIms = Flatten(TrainIms).astype('float64')
TestIms = Flatten(TestIms).astype('float64')

print(TrainIms.shape,TrainIms.dtype)
print(TrainLabels.shape,TrainLabels.dtype)
print(TestIms.shape)
print(TestLabels.shape)

## SAMPLING - In case we want to apply this to a subset of the data
TestS = 1000 #Size of test data
TrainS = 10000  #Size of training data
TestLabels = TestLabels[:TestS]
TestIms = TestIms[:TestS,:]
TrainLabels = TrainLabels[:TrainS]
TrainIms = TrainIms[:TrainS,:]

t0 = time.time()
#Calculating squared vector norms
TrainNorms = np.array([TrainIms[i,:]*TrainIms[i,:].T.toarray() for i in range(TrainIms.shape[0])]).reshape(-1,1)

def PredictandError(testims,testlabs):
    #This is not technically a distance - we are leaving out the Test squared norms because they are constant 
    #when determining a nearest neighbor.
    Distances = TrainNorms*np.ones(testims.shape[0]).T - 2*TrainIms*testims.T

    predictions = TrainLabels[np.argmin(Distances,axis=0)]

    error = 1 - np.mean(np.equal(predictions,testlabs))
    return(error*100)

BucketSize = 1000
errors = []
for i in range(0,len(TestLabels),BucketSize):
    errors.append(PredictandError(TestIms[i:(i+BucketSize)],TestLabels[i:(i+BucketSize)]))

t1 = time.time()
print('error:' + str(np.mean(errors))) #Since the buckets are equal size, we can average their errors.
print('Time taken:' + str(t1-t0))
