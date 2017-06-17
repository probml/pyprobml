#Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance

from scipy import sparse
import numpy as np

import struct
import os
import time

#This PULLMNIST function is from https://gist.github.com/akesling/5358964
def PullMNIST(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be \'testing\' or \'training\'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return {'labels':lbl,'images':img}
        
Training = PullMNIST("training",'/Data/MNIST')
TrainLabels = Training['labels']
TrainIms = Training['images'].astype('int32')

Testing = PullMNIST("testing",'/Data/MNIST')
TestLabels = Testing['labels']
TestIms = Testing['images'].astype('int32')

del Testing, Training

#Flattens images into sparse vectors. So we go from 3D to 2D image datasets.
def Flatten(Ims):
    return(sparse.csr_matrix(Ims.reshape(Ims.shape[0],-1)))

TrainIms = Flatten(TrainIms)
TestIms = Flatten(TestIms)

## SAMPLING - In case we want to apply this to a subset of the data
#TestS = 1000 #Size of test data
#TrainS = 10000  #Size of training data
#TestLabels = TestLabels[:TestS]
#TestIms = TestIms[:TestS,:]
#TrainLabels = TrainLabels[:TrainS]
#TrainIms = TrainIms[:TrainS,:]

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
