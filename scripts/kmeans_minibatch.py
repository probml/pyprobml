

# minibtach K-means clustering for MNIST
# Code is from chapter 9 of 
# https://github.com/ageron/handson-ml2 

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from sklearn.datasets import make_blobs

blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)

from timeit import timeit

K = 50
times = np.empty((K, 2))
inertias = np.empty((K, 2))
for k in range(1, K+1):
    kmeans_ = KMeans(n_clusters=k, random_state=42)
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    print("\r{}/{}".format(k, 100), end="")
    times[k-1, 0] = timeit("kmeans_.fit(X)", number=10, globals=globals())
    times[k-1, 1]  = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())
    inertias[k-1, 0] = kmeans_.inertia_
    inertias[k-1, 1] = minibatch_kmeans.inertia_
    
plt.figure(figsize=(10,5))

plt.subplot(121)
plt.plot(range(1, K+1), inertias[:, 0], "r--", label="K-Means")
plt.plot(range(1, K+1), inertias[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Distortion", fontsize=14)
plt.legend(fontsize=14)
#plt.axis([1, K, 0, K])

plt.subplot(122)
plt.plot(range(1, K+1), times[:, 0], "r--", label="K-Means")
plt.plot(range(1, K+1), times[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Training time (seconds)", fontsize=14)
#plt.axis([1, K, 0, 6])

plt.tight_layout()
pml.savefig("kmeans_minibatch.pdf", dpi=300)
plt.show()