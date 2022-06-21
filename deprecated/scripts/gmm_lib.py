# Library of Gaussian Mixture Models
# To-do: convert library into class
# Author: Gerardo Durán-Martín

import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def plot_mixtures(X, mu, pi, Sigma, r, step=0.01, cmap="viridis", ax=None):
    ax = ax if ax is not None else plt.subplots()[1]
    colors = ["tab:red", "tab:blue"]
    x0, y0 = X.min(axis=0)
    x1, y1 = X.max(axis=0)
    xx, yy = np.mgrid[x0:x1:step, y0:y1:step]
    zdom = np.c_[xx.ravel(), yy.ravel()]
    
    Norms = [multivariate_normal(mean=mui, cov=Sigmai)
             for mui, Sigmai in zip(mu, Sigma)]
    
    for Norm, color in zip(Norms, colors):
        density = Norm.pdf(zdom).reshape(xx.shape)
        ax.contour(xx, yy, density, levels=1,
                    colors=color, linewidths=5)
        
    ax.scatter(*X.T, alpha=0.7, c=r, cmap=cmap, s=10)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

def compute_responsibilities(k, pi, mu, sigma):
    Ns = [multivariate_normal(mean=mu_i, cov=Sigma_i) for mu_i, Sigma_i in zip(mu, sigma)]
    def respons(x):
        elements = [pi_i * Ni.pdf(x) for pi_i, Ni in zip(pi, Ns)]
        return elements[k] / np.sum(elements, axis=0)
    return respons


def e_step(pi, mu, Sigma):
    responsibilities = []
    for i, _ in enumerate(mu):
        resp_k = compute_responsibilities(i, pi, mu, Sigma)
        responsibilities.append(resp_k)
    return responsibilities


def m_step(X, responsibilities, S=None, eta=None):
    N, M = X.shape
    pi, mu, Sigma = [], [], []
    has_priors = eta is not None
    for resp_k in responsibilities:
        resp_k = resp_k(X)
        Nk = resp_k.sum()
        # mu_k
        mu_k = (resp_k[:, np.newaxis] * X).sum(axis=0) / Nk
        # Sigma_k
        dk = (X - mu_k)
        Sigma_k = resp_k[:, np.newaxis, np.newaxis] * np.einsum("ij, ik->ikj", dk, dk)
        Sigma_k = Sigma_k.sum(axis=0)
        if not has_priors:
            Sigma_k = Sigma_k / Nk
        else: 
            Sigma_k = (S + Sigma_k) / (eta + Nk + M + 2)

        # pi_k
        pi_k = Nk / N
        
        pi.append(pi_k)
        mu.append(mu_k)
        Sigma.append(Sigma_k)
    return pi, mu, Sigma


def gmm_log_likelihood(X, pi, mu, Sigma):
    likelihood =  0
    for pi_k, mu_k, Sigma_k in zip(pi, mu, Sigma):
        norm_k = multivariate_normal(mean=mu_k, cov=Sigma_k)
        likelihood += pi_k * norm_k.pdf(X)
    return np.log(likelihood).sum()


def apply_em(X, pi, mu, Sigma, threshold=1e-5, S=None, eta=None):
    r = compute_responsibilities(0, pi, mu, Sigma)(X)
    log_likelihood = gmm_log_likelihood(X, pi, mu, Sigma)
    hist_log_likelihood = [log_likelihood]
    hist_coeffs = [(pi, mu, Sigma)]
    hist_responsibilities = [r]

    while True:
        responsibilities = e_step(pi, mu, Sigma)
        pi, mu, Sigma = m_step(X, responsibilities, S, eta)
        log_likelihood = gmm_log_likelihood(X, pi, mu, Sigma)
        
        hist_coeffs.append((pi, mu, Sigma))
        hist_responsibilities.append(responsibilities[0](X))
        hist_log_likelihood.append(log_likelihood)
        
        if np.abs(hist_log_likelihood[-1] / hist_log_likelihood[-2] - 1) < threshold:
            break
        results = {
            "coeffs": hist_coeffs,
            "rvals": hist_responsibilities,
            "logl": hist_log_likelihood
        }
    return results
