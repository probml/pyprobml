# PCA applied to a 3d dataset projecting to 2d
# Compare to linear_autoencoder_pca_tf

import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

import tensorflow as tf
from tensorflow import keras

from sklearn.decomposition import PCA

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(4)

def generate_3d_data(m, w1=0.1, w2=0.3, noise=0.1):
    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty((m, 3))
    data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
    data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * np.random.randn(m)
    return data

X_train = generate_3d_data(60)
X_train = X_train - X_train.mean(axis=0, keepdims=0)

pca = PCA(n_components=2)
mu = np.mean(X_train, axis=0)
Xc = X_train - mu # center the data
pca.fit(Xc)
W = pca.components_.T # D*K
Z = np.dot(Xc, W) # N * K latent scores
Xrecon = np.dot(Z, W.T) + mu # N*D

fig = plt.figure(figsize=(4,3))
plt.plot(Z[:,0], Z[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
save_fig("pca-demo-2d.pdf")
plt.show()


# Plot original data in 3d
X = X_train
fig = plt.figure().gca(projection='3d')
fig.scatter(X[:,0], X[:,1], X[:,2], s=50, marker='o')
save_fig("pca-demo-3d.pdf")
plt.show()
