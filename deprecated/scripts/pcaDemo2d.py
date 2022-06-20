import superimport

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import pyprobml_utils as pml

np.random.seed(42)

# Generate Data
n = 9
var = 3
corr = .5

cov_mat = [[var, corr * var], [corr * var, var]]
X = np.random.multivariate_normal([0, 0], cov_mat, n)

# Project Data onto PCA line
subspace_dim = 1
pca = PCA(subspace_dim)
X_reconstr = pca.inverse_transform(pca.fit_transform(X))

# Create figure and save.
fig, ax = plt.subplots(figsize=(5, 5))

# Plot raw data
ax.scatter(X[:, 0], X[:, 1], marker='o', facecolor='none', edgecolor='red')
X_mean = np.mean(X, axis=0)
ax.scatter(X_mean[0], X_mean[1], facecolor='red')

# Plot PCA line
low_point = X_mean - 10 * pca.components_.reshape(-1)
high_point = X_mean + 10 * pca.components_.reshape(-1)
ax.plot([low_point[0], high_point[0]], [low_point[1], high_point[1]], color='magenta')
ax.set_ylim(-5, 5)
ax.set_xlim(-5, 5)

# Plot projected points
ax.scatter(X_reconstr[:, 0], X_reconstr[:, 1], marker='x')

# Plot projection lines
for (xi1, xi2), (xi1_rec, xi2_rec) in zip(X, X_reconstr):
    ax.plot([xi1, xi1_rec], [xi2, xi2_rec], color='blue')


pml.savefig("pcaDemo2dProjection.pdf")

plt.show()