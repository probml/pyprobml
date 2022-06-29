# Mean shift for clustering 2d data
# From 
#https://github.com/log0/build-your-own-meanshift/blob/master/Meanshift%20In%202D.ipynb
#http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/

import superimport

import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
#%matplotlib inline
#pylab.rcParams['figure.figsize'] = 16, 12

import os
figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

np.random.seed(42)
original_X, X_shapes = make_blobs(100, 2, centers=4, cluster_std=1.3)
print(original_X.shape)
plt.plot(original_X[:,0], original_X[:,1], 'bo', markersize = 10)

def euclid_distance(x, xi):
    return np.sqrt(np.sum((x - xi)**2))

def neighbourhood_points(X, x_centroid, distance = 5):
    eligible_X = []
    for x in X:
        distance_between = euclid_distance(x, x_centroid)
        # print('Evaluating: [%s vs %s] yield dist=%.2f' % (x, x_centroid, distance_between))
        if distance_between <= distance:
            eligible_X.append(x)
    return eligible_X

def gaussian_kernel(distance, bandwidth):
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
    return val

look_distance = 5 #  # How far to look for neighbours.
#kernel_bandwidth = 2 # # Kernel parameter.

quantile = 0.1 # good results
#quantile = 0.3 # default value, gives bad results
kernel_bandwidth = estimate_bandwidth(original_X, quantile=quantile)
print("quantile {}, bandwidth {}".format(quantile, kernel_bandwidth))
bandwidth_str = '{}'.format(int(kernel_bandwidth*10))
quantile_str = '{}'.format(int(quantile*10))

X = np.copy(original_X)
# print('Initial X: ', X)

past_X = []
n_iterations = 5
for it in range(n_iterations):
    # print('Iteration [%d]' % (it))    

    for i, x in enumerate(X):
        ### Step 1. For each datapoint x ∈ X, find the neighbouring points N(x) of x.
        neighbours = neighbourhood_points(X, x, look_distance)
        # print('[%s] has neighbours [%d]' % (x, len(neighbours)))
        
        ### Step 2. For each datapoint x ∈ X, calculate the mean shift m(x).
        numerator = 0
        denominator = 0
        for neighbour in neighbours:
            distance = euclid_distance(neighbour, x)
            weight = gaussian_kernel(distance, kernel_bandwidth)
            numerator += (weight * neighbour)
            denominator += weight
        
        new_x = numerator / denominator
        
        ### Step 3. For each datapoint x ∈ X, update x ← m(x).
        X[i] = new_x
    
    # print('New X: ', X)
    past_X.append(np.copy(X))


plt.figure()
plt.title('Initial state, quantile {:0.2f}, bandwidth {:0.2f}'.format(quantile, kernel_bandwidth))
plt.plot(original_X[:,0], original_X[:,1], 'bo')
plt.plot(original_X[:,0], original_X[:,1], 'ro')
save_fig('meanshift-cluster-init-Q{}.pdf'.format(quantile_str))
plt.show()

for i in range(n_iterations):
    plt.figure()
    plt.title('Iteration: %d' % i)
    plt.plot(original_X[:,0], original_X[:,1], 'bo')
    plt.plot(past_X[i][:,0], past_X[i][:,1], 'ro')
    save_fig('meanshift-cluster-iter{}-Q{}.pdf'.format(i, quantile_str))
    plt.show()
    