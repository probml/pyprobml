# Comparing ICA and PCA on data from a 2d uniform distribution
# Author : Aleyna Kara
# This file is based on https://github.com/probml/pmtk3/blob/master/demos/icaDemoUniform.m

import superimport

from sklearn.decomposition import PCA, FastICA
import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

def plot_samples(S, title, file_name):
  min_x, max_x = -4, 4
  min_y, max_y = -3, 3
  plt.scatter(S[:, 0], S[:, 1], marker='o',s=16)
  plt.hlines(0, min_x, max_x, linewidth=2)
  plt.vlines(0, min_y, max_y, linewidth=2)
  plt.xlim(min_x, max_x)
  plt.ylim(min_y, max_y)
  plt.title(title)
  pml.savefig(f'{file_name}.pdf')
  plt.show()

np.random.seed(2)
N = 100
A = np.array([[2,3],[2,1]])* 0.3 # Mixing matrix

S_uni = (np.random.rand(N, 2)* 2 - 1)* np.sqrt(3)
X_uni = S_uni @ A.T

pca = PCA(whiten=True)
S_pca = pca.fit(X_uni).transform(X_uni)

ica = FastICA()
S_ica = ica.fit_transform(X_uni)
S_ica /= S_ica.std(axis=0)

plot_samples(S_uni, 'Uniform Data', 'ica-uniform-source')

plot_samples(X_uni, 'Uniform Data after Linear Mixing', 'ica-uniform-mixed')

plot_samples(S_pca, 'PCA Applied to Mixed Data from Uniform Source', 'ica-uniform-PCA')

plot_samples(S_ica, 'ICA Applied to Mixed Data from Uniform Source', 'ica-uniform-ICA')