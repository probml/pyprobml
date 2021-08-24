# Demonstrate failure of MLE for GMMs in high-D case, whereas MAP works
# Based on: https://github.com/probml/pmtk3/blob/master/demos/mixGaussMLvsMAP.m
# Author: Gerardo Durán-Martín, Aleyna Kara(@karalleyna)

import superimport

import jax.numpy as jnp
from jax.random import PRNGKey, split, normal
from jax import tree_multimap

import matplotlib.pyplot as plt
from numpy.linalg import cholesky
from mix_gauss_lib import GMM

import warnings
warnings.filterwarnings("ignore")

def fill_cov(S, dim):
    m, m = S.shape
    S_eye = jnp.identity(dim - m)
    S_fill = jnp.zeros((m, dim - m))
    S_fill_left = jnp.vstack((S_fill, S_eye))
    S_final = jnp.vstack((S, S_fill.T))
    S_final = jnp.hstack((S_final, S_fill_left))
    return S_final

def init_sigma(sigma_bases, test_dims):
  sigmas = []
  for dim in test_dims:
    Sigma1_base, Sigma2_base, Sigma3_base = sigma_bases
    Sigma1 = fill_cov(Sigma1_base, dim)
    Sigma2 = fill_cov(Sigma2_base, dim)
    Sigma3 = fill_cov(Sigma3_base, dim)

    Sigma = jnp.stack((Sigma1, Sigma2, Sigma3), axis=0)
    sigmas.append(Sigma)
  return sigmas

def init_samples(mu_base, sigmas, test_dims, keys, n_samples):
  res = []
  for (Sigma, dim), key in zip(zip(sigmas, test_dims), keys):
    mu = jnp.hstack([mu_base, jnp.zeros((3, dim - 2))])
    R = cholesky(Sigma)
    samples = jnp.ones((n_samples, 1, 1)) * mu[None, ...]

    noise = normal(key, (n_samples, dim))
    noise = jnp.einsum("kjm,nj->nkm", R, noise)

    samples = samples + noise
    samples = samples.reshape(-1, dim)
    res.append(samples)
  return res

def attempt_em_fit(X, Sigma, pi, dim, k=3, n_attempts=3):
    N, M = X.shape
    eta = M + 2
    n_success_ml = 0
    n_success_map = 0
    S = X.std(axis=0)
    S = jnp.diag(S ** 2) / (k ** (1 / M))
    for n in range(n_attempts):
        mu = normal(PRNGKey(n), (k, dim))
        try:
            gmm = GMM(pi, mu, Sigma)
            gmm.fit_em(X, num_of_iters=5)
            n_success_ml += 1
        except Exception as E:
            print(str(E))
        try:
            gmm = GMM(pi, mu, Sigma)
            gmm.fit_em(X, num_of_iters=5, S=S, eta=eta)
            n_success_map += 1
        except Exception as E:
            print(str(E))
    pct_ml = n_success_ml / n_attempts
    pct_map = n_success_map / n_attempts
    return [1-pct_ml, 1-pct_map]

rng_key = PRNGKey(0)
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

n_comps = 3
pi = jnp.ones((n_comps, )) / n_comps
hist_ml, hist_map = [], []

test_dims = jnp.arange(10, 60, 10)
keys = split(rng_key, 10)

n_samples = 150
mu_base = jnp.array([[-1, 1], [1, -1], [3, -1]])

Sigma1_base = jnp.array([[1, -0.7], [-0.7, 1]])
Sigma2_base = jnp.array([[1, 0.7], [0.7, 1]])
Sigma3_base = jnp.array([[1, 0.9], [0.9, 1]])
sigmas = init_sigma((Sigma1_base, Sigma2_base, Sigma3_base), test_dims)

samples = init_samples(mu_base, sigmas, test_dims, keys, n_samples)
hist_ml, hist_map = jnp.array(tree_multimap(lambda X, Sigma, dim: attempt_em_fit(X, Sigma, pi, dim),
                                            samples, sigmas, test_dims.tolist())).T

fig, ax = plt.subplots()
ax.plot(test_dims, hist_ml, c="tab:red", marker="o", label="MLE")
ax.plot(test_dims, hist_map, c="black", marker="o", linestyle="--", label="MAP")
ax.set_xlabel("dimensionality")
ax.set_ylabel("fraction of times EM for GMM fails")
ax.legend()
plt.show()