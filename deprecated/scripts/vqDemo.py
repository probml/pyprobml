# Vector Quantization Demo
# Author: Animesh Gupta

# Use racoon face image
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.face.html

import superimport

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn import cluster
import pyprobml_utils as pml

try:  # SciPy >= 0.16 have face in misc
    from scipy.misc import face
    face = face(gray=True)
except ImportError:
    face = sp.face(gray=True)

n_clusters = [2,4]
np.random.seed(0)

X = face.reshape((-1, 1))  # We need an (n_sample, n_feature) array
for n_cluster in n_clusters:
    k_means = cluster.KMeans(n_clusters=n_cluster, n_init=4)
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    
    # create an array from labels and values
    face_compressed = np.choose(labels, values)
    face_compressed.shape = face.shape
    
    vmin = face.min()
    vmax = face.max()
    
    # compressed face
    plt.figure(figsize=(4,4))
    plt.title(f'K = {n_cluster}')
    plt.imshow(face_compressed, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
    pml.savefig(f"vectorQuantization_{n_cluster}.pdf", dpi=300)
    plt.show()
 