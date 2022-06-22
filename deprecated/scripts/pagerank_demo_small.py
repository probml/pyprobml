'''
Finds the stationary distributions for the graph consisting of 6 nodes and one for Harvard500. Then, compares
the results found by matrix inversion and power method. Plots the bar plots.
Author: Cleve Moler, Aleyna Kara
This file is converted to Python from https://github.com/probml/pmtk3/blob/master/demos/pagerankDemoPmtk.m
'''

import superimport

import numpy as np
from pagerank_power_method_sparse import pagerank_power_method_sparse
import matplotlib.pyplot as plt
import pyprobml_utils as pml


'''
Computes the stationary distribution of G given the probability p  
    Arguments:
        G : Adjacency matrix representing the link structure. The  ijth element being one 
        means that there exists a link from j to i. 
        p : The probability of a link to be followed.
'''
def matrix_inversion_method(G, p):
  # Graph should be directed acylic
  G = G - np.diag(G.diagonal())
  n, _ = G.shape
  out_degree = np.squeeze(np.asarray(np.sum(G, axis=0)))
  # Scales column sums to be 1 (or 0 where there are no out links).
  D = np.diag(np.where(out_degree!= 0, 1/out_degree, 0))
  # Solves (I - p * G @ D) @ x = e
  e = np.ones(n)
  x = np.linalg.solve(np.eye(n) - p * G @ D, e)

  # Normalizes so that prob(x) = sum(x) = 1.
  x = x / np.sum(x)
  return x

# Link structure of small web
G = np.array([[0, 0, 0, 1, 0, 1],[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0],[0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
p = 0.85
pi_matinv = matrix_inversion_method(G, p)
pi_sparse = pagerank_power_method_sparse(G, p)[0]

try:
    assert np.allclose(pi_matinv, pi_sparse, atol=1e-4)
except AssertionError:
    print('pi_matinv != pi_spare')

n, _ = G.shape
labels = [f'$X_{i}$' for i in range(1, n+1)]
x = range(n)
plt.bar(x, pi_matinv)
plt.xticks(x, labels)
pml.savefig('pagerank-small.pdf')
plt.show()
