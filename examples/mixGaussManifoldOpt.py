# Fit a 2d MOG model using optimization over a Riemannian manifold.
# as described in this paper http://arxiv.org/pdf/1506.07677v1.pdf
# Code is slightly modified from
# https://pymanopt.github.io/MoG.html

import autograd.numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
#%matplotlib inline
from autograd.scipy.misc import logsumexp
from pymanopt.manifolds import Product, Euclidean, PositiveDefinite
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions

# Number of data
N = 1000

# Dimension of data
D = 2

# Number of clusters
K = 3

# True model parameters
pi = [0.1, 0.6, 0.3]
mu = [np.array([-4, 1]), np.array([0, 0]), np.array([2, -1])]
Sigma = [np.array([[3, 0],[0, 1]]), np.array([[1, 1.], [1, 3]]), .5 * np.eye(2)]

# Sample some data
components = np.random.choice(K, size=N, p=pi)
samples = np.zeros((N, D))
for k in range(K):
    # indices of current component in X
    indices = (k == components)
    # number of those occurrences
    n_k = indices.sum()
    if n_k > 0:
        samples[indices] = np.random.multivariate_normal(mu[k], Sigma[k], n_k)

# Plot the data
colors = ['r', 'g', 'b', 'c', 'm']
for i in range(K):
    indices = (i == components)
    plt.scatter(samples[indices, 0], samples[indices, 1], alpha=.4, color=colors[i%K])
plt.axis('equal')
plt.show()



# (1) Instantiate the manifold
manifold = Product([PositiveDefinite(D+1, k=K), Euclidean(K-1)])

# (2) Define cost function
# The parameters must be contained in a list theta.
def cost(theta):
    # Unpack parameters
    nu = np.concatenate([theta[1], [0]], axis=0)
    
    S = theta[0]
    logdetS = np.expand_dims(np.linalg.slogdet(S)[1], 1)
    y = np.concatenate([samples.T, np.ones((1, N))], axis=0)

    # Calculate log_q
    y = np.expand_dims(y, 0)
    
    # 'Probability' of y belonging to each cluster
    log_q = -0.5 * (np.sum(y * np.linalg.solve(S, y), axis=1) + logdetS)

    alpha = np.exp(nu)
    alpha = alpha / np.sum(alpha)
    alpha = np.expand_dims(alpha, 1)
    
    loglikvec = logsumexp(np.log(alpha) + log_q, axis=0)
    return -np.sum(loglikvec)

problem = Problem(manifold=manifold, cost=cost, verbosity=1)

# (3) Instantiate a Pymanopt solver
#solver = TrustRegions()
solver = SteepestDescent(logverbosity=1)

# let Pymanopt do the rest
Xopt, optlog = solver.solve(problem)
print optlog

# Inspect results
mu_hat = Xopt[0]
Sigma_hat = Xopt[1]
for k in range(K):
  mu_est = Xopt[0][k][0:2, 2:3]
  Sigma_est = Xopt[0][k][:2, :2] - mu_est.dot(mu_est.T)
  print 'k = {}'.format(k)
  print 'true mu {}, est {}'.format(mu[k], np.ravel(mu_est))
  
pihat = np.exp(np.concatenate([Xopt[1], [0]], axis=0))
pihat = pihat / np.sum(pihat)
print 'true pi {}, est {}'.format(pi, pihat)

