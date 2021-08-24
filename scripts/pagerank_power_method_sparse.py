# Implements the power method without using any matrix multiplications, i.e. Monte Carlo approximation to the sum implied by v = Mv
# Author : Cleve Moler, Aleyna Kara
# This function is the Python implementation of  https://github.com/probml/pmtk3/blob/master/demos/pagerankpow.m

import superimport

import numpy as np
from scipy.sparse import coo_matrix


'''
Returns the PageRank of the graph G and the number of iterations
Arguments:
    G : Adjacency matrix representing the link structure
    p : The probability of a link to be followed.
'''
def pagerank_power_method_sparse(G, p):
    # Link structure
    n, _ = G.shape

    # The fastest way to iterate columns of csc_matrix
    cg = coo_matrix(G)
    L = [[] for i in range(n)]
    for i, j in zip(cg.row, cg.col):
        L[j].append(i)
    c = [len(L[j]) for j in range(n)]

    # Power method
    delta, n_iterations = (1 - p) / n, 0
    x, z = np.ones((n, 1)) / n, np.zeros((n, 1))

    while np.max(np.abs(x - z)) > 1e-4:
        z = x
        x = np.zeros(n)
        for j in range(n):
            if not c[j]:
                x = x + z[j] / n
            else:
                x[L[j]] = x[L[j]] + z[j] / c[j]
        x = p * x + delta
        n_iterations += 1

    return x, n_iterations