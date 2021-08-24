import superimport

import numpy as np
from scipy.stats import multivariate_normal as mvn

def isposdef(A):
    try:
        _ = np.linalg.cholesky(A)
        return True
    except:
        return False
    
# test
np.random.seed(42)
D = 5
A = np.random.randn(D, D)
assert not(isposdef(A))
assert isposdef(np.dot(A, A.T))
    

def sample_mvn(mu, Sigma, N):
    L = np.linalg.cholesky(Sigma)
    D = len(mu)
    Z = np.random.randn(N, D)
    X = np.dot(Z, L.T) + np.reshape(mu, (-1,D))
    return X

# test
D = 5
np.random.seed(42)
mu = np.random.randn(D)
A = np.random.randn(D,D)
Sigma = np.dot(A, A.T)
N = 1000
X = sample_mvn(mu, Sigma, N)
C = np.cov(X, rowvar=False)
assert np.allclose(C, Sigma, 1e-0)

dist = mvn(mu, Sigma)
N = 1000
X = dist.rvs(size=N)
C = np.cov(X, rowvar=False)
assert np.allclose(C, Sigma, 1e-0)
    