# Illustrate the powr method for computing largest eigenvector

import superimport

import numpy as np
from numpy.linalg import norm

np.random.seed(0)

def power_method(A, max_iter=100, tol=1e-5):
    n = np.shape(A)[0]
    u = np.random.rand(n)
    converged = False
    iter = 0
    while (not converged) and (iter < max_iter):
        old_u = u
        u = np.dot(A, u)
        u = u / norm(u)
        lam = np.dot(u, np.dot(A, u))
        converged = (norm(u - old_u) < tol)
        iter += 1
    return lam, u

X = np.random.randn(10, 5)
A = np.dot(X.T, X)
lam, u = power_method(A)

evals, evecs = np.linalg.eig(A)
idx = np.argsort(np.abs(evals))[::-1] # largest first
evals = evals[idx]
evecs = evecs[:,idx]

tol = 1e-3
assert np.allclose(evecs[:,0], u, tol)
assert np.allclose(evals[0], lam, tol)
