# centering demo
import superimport

import numpy as np

np.random.seed(0)
N = 3;
D = 2;
X = np.random.rand(D,N);

e=np.ones(N)
H=np.eye(N) - (1/N)*e*e.T
z=np.diag(np.dot(X.T, X)) # vector of norms oer data point
Z=np.outer(z,e)
D = Z - 2*X.T @ X + Z.T # pairwise distances

A = H @ D @ H
m  = (1/N)*np.dot(X,e)
XX=X - np.outer(m,e)
AA = -2*XX.T @ XX

assert np.allclose(A, AA)
assert np.allclose(H @ Z @ H, np.zeros((N,N)))
