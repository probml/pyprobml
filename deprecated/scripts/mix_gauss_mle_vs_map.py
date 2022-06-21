# Demonstrate failure of MLE for GMMs in high-D case, whereas MAP works
# Based on: https://github.com/probml/pmtk3/blob/master/demos/mixGaussMLvsMAP.m

# Author: Gerardo Durán-Martín

import superimport

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, seed
from numpy.linalg import cholesky, LinAlgError
import pyprobml_utils as pml
import gmm_lib

def fill_cov(S, dim):
    m, m = S.shape
    S_eye = np.identity(dim - m)
    S_fill = np.zeros((m, dim - m))
    S_fill_left = np.r_[S_fill, S_eye]
    S_final = np.r_[S, S_fill.T]
    S_final = np.c_[S_final, S_fill_left]

    return S_final

def attempt_em_fit(X, k, pi, Sigma, n_attempts=5):
    N, M = X.shape
    eta = M + 2
    n_success_ml = 0
    n_success_map = 0
    S = X.std(axis=0)
    S = np.diag(S ** 2) / (k ** (1 / M))
    for n in range(n_attempts):
        mu = randn(k, dim)
        try:
            gmm_lib.apply_em(X, pi, mu, Sigma)
            n_success_ml += 1
        except LinAlgError:
            pass
        try:
            gmm_lib.apply_em(X, pi, mu, Sigma, S=S, eta=eta)
            n_success_map += 1
        except LinAlgError:
            pass
    pct_ml = n_success_ml / n_attempts
    pct_map = n_success_map / n_attempts
    return pct_ml, pct_map

if __name__ == "__main__":
    seed(314)
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    pi = np.ones(3) / 3
    hist_ml, hist_map = [], []
    test_dims = np.arange(10, 110, 10)
    n_samples = 150

    for dim in test_dims:
        mu_base = np.array([[-1, 1], [1, -1], [3, -1]])
        Sigma1_base = np.array([[1, -0.7], [-0.7, 1]])
        Sigma2_base = np.array([[1, 0.7], [0.7, 1]])
        Sigma3_base = np.array([[1, 0.9], [0.9, 1]])

        mu = np.c_[mu_base, np.zeros((3, dim - 2))]
        Sigma1 = fill_cov(Sigma1_base, dim)
        Sigma2 = fill_cov(Sigma2_base, dim)
        Sigma3 = fill_cov(Sigma3_base, dim)

        Sigma = np.stack((Sigma1, Sigma2, Sigma3), axis=0)
        R = cholesky(Sigma)
        samples = np.ones((n_samples, 1, 1)) * mu[None, ...]

        noise = randn(n_samples, dim)
        noise = np.einsum("kjm,nj->nkm", R, noise)

        samples = samples + noise
        samples = samples.reshape(-1, dim)

        ml, map = attempt_em_fit(samples, 3, pi, Sigma)
        hist_ml.append(1 - ml)
        hist_map.append(1 - map)

    fig, ax = plt.subplots()
    ax.plot(test_dims, hist_ml, c="tab:red", marker="o", label="MLE")
    ax.plot(test_dims, hist_map, c="black", marker="o", linestyle="--", label="MAP")
    ax.set_xlabel("dimensionality")
    ax.set_ylabel("fraction of times EM for GMM fails")
    ax.legend()
    pml.savefig("gmm_mle_vs_map.pdf")
    plt.show()
