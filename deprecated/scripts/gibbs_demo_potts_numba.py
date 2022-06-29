# -*- coding: utf-8 -*-
"""
Author: Ang Ming Liang
"""

import superimport

import numpy as np
from numba import njit
import scipy.linalg as la
#from tqdm.notebook import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import pyprobml_utils as pml

np.random.seed(1)

K=10
size = 64

Kmatrix = np.eye(K)

# Calculate adjecency matrix of the entire grid
offdi = la.toeplitz([0,1]+[0]*(size-2))
I = np.eye(size)
A = (np.kron(offdi,I) + np.kron(I,offdi)).astype(int)
A_id = A*np.arange(size*size)

@njit
def energy(state_mat, pos, Jvalue):
  neigh_states = state_mat[pos][A_id[pos] > 0]
  neigh_states_one_hot = Kmatrix[neigh_states]
  E = Jvalue*np.sum(neigh_states_one_hot, axis=0)
  return E

@njit
def sample(logits):
  # Gumbel trick to sample efficiently from the categorical distribution
  # See this for more info : https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html
  u = np.random.random(logits.shape)
  return np.argmax(logits - np.log(-np.log(u)))

@njit
def gibbs_sampler(jvalue, niter=100):
  X = np.random.randint(0, K,  size=(size*size, ))
  state_mat = X*A

  for iter in range(niter):
    for x in range(size):
      for y in range(size):
        pos = x + y*size

        E = energy(state_mat, pos, jvalue)
        rv = sample(E)
        
        X[pos] = rv
        state_mat[:, pos][A_id[pos] > 0] = rv

  return X.reshape((size, size))

Jvals = [1.40, 1.43, 1.46]

#fig, axs = plt.subplots(1, len(Jvals), figsize=(8, 8))
for t, j in tqdm(enumerate(Jvals)):
  fig, ax  = plt.subplots()
  sample = gibbs_sampler(j, niter=400)
  ax.imshow(sample, cmap='Accent')
  ax.set_title(f"J= {j}")
  pml.savefig(f'gibbsDemoPotts{t}.pdf')
  plt.show()
