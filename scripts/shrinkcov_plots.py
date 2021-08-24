# Based on https://github.com/probml/pmtk3/blob/master/demos/shrinkcovDemo.m
# Converted by John Fearns - jdf22@infradead.org
# Demo of the quality of shrinkage estimation of a covariance matrix

import superimport

import numpy as np
import matplotlib.pyplot as plt
from pyprobml_utils import save_fig

# Ensure stochastic reproducibility.
np.random.seed(42)

# Dimensions
D = 50

# A function that generates a random covariance matrix (and its inverse) with the given condition number
# and with the given first axis a (unnormalised). Returns (Sigma, Lambda).
def covcond(cond_number, a):
    # Use the Householder trick to generate a basis Z whose first axis points in the direction of a.
    a = np.copy(a)
    a[0] = a[0] + np.sign(a[1]) * np.linalg.norm(a)
    Z = np.eye(len(a)) - 2*np.matmul(a.reshape(-1,1), a.reshape(1,-1)) / np.linalg.norm(a)**2
    
    e = np.flip(np.sort(1.0 / np.linspace(cond_number, 1, len(a))))
    Sigma = np.matmul(np.matmul(Z, np.diag(e)), Z.transpose())
    Lambda = np.matmul(np.matmul(Z, np.diag(1/e)), Z.transpose())
    
    return (Sigma, Lambda)

# Generate a random covariance matrix with condition number 10
# and record its eigenvalues.
cond_number = 10
a = np.random.randn(D,1)
Sigma, Lambda = covcond(cond_number, a)
evals_true = np.flip(np.sort(np.linalg.eig(Sigma)[0]))

# Fractions of D
fractions = [2, 1, 1/2]

# Make way for 3 plots arranged in a suitable 1x3 grid.
figure, axes = plt.subplots(1, 3, figsize=(16, 4.75))
linewidth=3

for i in range(len(fractions)):
    # How many samples?
    n = int(fractions[i]*D)
    
    # Sample using Sigma.
    X = np.random.multivariate_normal(np.zeros(D), Sigma, n)
    
    # We set bias=True to get the MLE estimate of Sigma.
    S_mle = np.cov(X, rowvar=False, bias=True)
    evals_mle = np.flip(np.sort(np.linalg.eig(S_mle)[0]))
    
    # Get the MAP estimate of Sigma.
    _lambda = 0.9
    S_shrink = _lambda*np.diag(np.diag(S_mle)) + (1-_lambda)*S_mle
    evals_shrink = np.flip(np.sort(np.linalg.eig(S_shrink)[0]))
    
    # Plot the eigevalues of the estimates and the true covariance matrix.
    ax = axes[i]
    ax.set_ylim(0, 1.5)
    ax.set_ylabel('eigenvalue')
    ax.plot(np.arange(1,D+1), evals_true, color='black', linestyle='-', linewidth=linewidth,
           label=r'true, $\kappa$={:.2f}'.format(np.linalg.cond(Sigma)))
    ax.plot(np.arange(1,D+1), evals_mle, color='blue', linestyle=':', linewidth=linewidth,
           label=r'MLE, $\kappa$={:.2e}'.format(np.linalg.cond(S_mle)))
    ax.plot(np.arange(1,D+1), evals_shrink, color='red', linestyle='-.', linewidth=linewidth,
           label=r'MAP, $\kappa$={:.2f}'.format(np.linalg.cond(S_shrink)))
    ax.set_title('N={}, D={}'.format(n, D))
    ax.legend(loc='upper right')
    
save_fig('covShrinkDemo.pdf')
plt.show()