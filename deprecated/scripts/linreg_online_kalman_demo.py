# Online Bayesian linear regression using Kalman Filter
# Based on: https://github.com/probml/pmtk3/blob/master/demos/linregOnlineDemoKalman.m
# Author: Gerardo Durán-Martín (@gerdm)

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from numpy.linalg import inv


def kf_linreg(X, y, R, mu0, Sigma0, F, Q):
    """
    Online estimation of a linear regression
    using Kalman Filters

    Parameters
    ----------
    X: array(n_obs, dimension)
        Matrix of features
    y: array(n_obs,)
        Array of observations
    Q: float
        Known variance
    mu0: array(dimension)
        Prior mean
    Sigma0: array(dimesion, dimension)
        Prior covariance matrix

    Returns
    -------
    * array(n_obs, dimension)
        Online estimation of parameters
    * array(n_obs, dimension, dimension)
        Online estimation of uncertainty
    """
    n_obs, dim = X.shape
    mu_hist = np.zeros((n_obs, dim))
    Sigma_hist = np.zeros((n_obs, dim, dim))

    I = np.eye(dim)
    Sigma_t = Sigma0.copy()
    mu_t = mu0.copy()

    mu_hist[0] = mu_t
    Sigma_hist[0] = Sigma_t
    for t in range(n_obs):
        xt, yt = X[t], y[t]

        Ht = xt[None, :]
        Sigma_t = F @ Sigma_t @ F.T + Q
        St = Ht @ Sigma_t @ Ht.T + R
        Kt = Sigma_t @ Ht.T @ inv(St).squeeze(-1)
        mu_t = F @ mu_t + Kt * (yt - Ht @ F @ mu_t)
        Sigma_t = (I - Kt[:, None] @ Ht) @ Sigma_t

        mu_hist[t] = mu_t
        Sigma_hist[t] = Sigma_t

    return mu_hist, Sigma_hist


def posterior_lreg(X, y, R, mu0, Sigma0):
    """
    Compute mean and covariance matrix of a
    Bayesian Linear regression

    Parameters
    ----------
    X: array(n_obs, dimension)
        Matrix of features
    y: array(n_obs,)
        Array of observations
    R: float
        Known variance
    mu0: array(dimension)
        Prior mean
    Sigma0: array(dimesion, dimension)
        Prior covariance matrix

    Returns
    -------
    * array(dimension)
        Posterior mean
    * array(n_obs, dimension, dimension)
        Posterior covariance matrix
    """
    Sn_bayes_inv = inv(Sigma0) + X.T @ X / R
    Sn_bayes = inv(Sn_bayes_inv)
    mn_bayes = Sn_bayes @ (inv(Sigma0) @ mu0 + X.T @ y / R)

    return mn_bayes, Sn_bayes


if __name__ == "__main__":
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    n_obs = 21
    timesteps = np.arange(n_obs)
    x = np.linspace(0, 20, n_obs)
    X = np.c_[np.ones(n_obs), x]
    F = np.eye(2)
    mu0 = np.zeros(2)
    Sigma0 = np.eye(2) * 10.

    Q, R = 0, 1
    # Data from original matlab example
    y = np.array([2.4865, -0.3033, -4.0531, -4.3359, -6.1742, -5.604, -3.5069, -2.3257, -4.6377,
                  -0.2327, -1.9858, 1.0284, -2.264, -0.4508, 1.1672, 6.6524, 4.1452, 5.2677, 6.3403, 9.6264, 14.7842])

    # Online estimation
    mu_hist, Sigma_hist = kf_linreg(X, y, R, mu0, Sigma0, F, Q)
    kf_var = Sigma_hist[-1, [0, 1], [0, 1]]
    w0_hist, w1_hist = mu_hist.T
    w0_err, w1_err = np.sqrt(Sigma_hist[:, [0, 1], [0, 1]].T)

    # Offline estimation
    (w0_post, w1_post), Sigma_post = posterior_lreg(X, y, R, mu0, Sigma0)
    w0_std, w1_std = np.sqrt(Sigma_post[[0, 1], [0, 1]])

    # Asserting values for means and variance
    assert np.allclose(w0_hist[-1], w0_post)
    assert np.allclose(w1_hist[-1], w1_post)
    assert np.allclose(w0_err[-1], w0_std)
    assert np.allclose(w1_err[-1], w1_std)

    fig, ax = plt.subplots()
    ax.errorbar(timesteps, w0_hist, w0_err, fmt="-o", label="$w_0$", color="black", fillstyle="none")
    ax.errorbar(timesteps, w1_hist, w1_err, fmt="-o", label="$w_1$", color="tab:red")

    ax.axhline(y=w0_post, c="black", label="$w_0$ batch")
    ax.axhline(y=w1_post, c="tab:red", linestyle="--", label="$w_1$ batch")

    ax.fill_between(timesteps, w0_post - w0_std, w0_post + w0_std, color="black", alpha=0.4)
    ax.fill_between(timesteps, w1_post - w1_std, w1_post + w1_std, color="tab:red", alpha=0.4)

    plt.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("weights")
    ax.set_ylim(-8, 4)
    ax.set_xlim(-0.5, n_obs)
    pml.savefig("linreg-online-kalman.pdf")
    plt.show()
