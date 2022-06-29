
# utility functions for multivariate Gaussians
# author: murphyk@

import superimport

import numpy as np


def is_pos_def(x):
    j = np.linalg.eigvals(x)
    return np.all(j > 0)


def gauss_sample(mu, sigma, n):
    a = np.linalg.cholesky(sigma)
    z = np.random.randn(len(mu), n)
    k = np.dot(a, z)
    return np.transpose(mu + k)


def gauss_condition(mu, sigma, visible_nodes, visible_values):
     d = len(mu)
     j = np.array(range(d))
     v = visible_nodes.reshape(len(visible_nodes))
     h = np.setdiff1d(j, v)
     if len(h)==0:
         mugivh = np.array([])
         sigivh = np.array([])
     elif len(v) == 0:
         mugivh = mu
         sigivh = sigma
     else:
         ndx_hh = np.ix_(h, h)
         sigma_hh = sigma[ndx_hh]
         ndx_hv = np.ix_(h, v)
         sigma_hv = sigma[ndx_hv]
         ndx_vv = np.ix_(v, v)
         sigma_vv = sigma[ndx_vv]
         sigma_vv_inv = np.linalg.inv(sigma_vv)
         visible_values_len = len(visible_values)
         mugivh = mu[h] + np.dot(sigma_hv, np.dot(sigma_vv_inv, (visible_values.reshape((visible_values_len,1))-mu[v].reshape((visible_values_len, 1)))))
         sigivh = sigma_hh - np.dot(sigma_hv, np.dot(sigma_vv_inv, np.transpose(sigma_hv)))
     return mugivh, sigivh


def gauss_impute(mu, sigma, x):
    n_data, data_dim = x.shape
    x_imputed = np.copy(x)
    for i in range(n_data):
        hidden_nodes = np.argwhere(np.isnan(x[i, :]))
        visible_nodes = np.argwhere(~np.isnan(x[i, :]))
        visible_values = np.zeros(len(visible_nodes))
        for tc, h in enumerate(visible_nodes):
            visible_values[tc] = x[i, h]
        mu_hgv, sigma_hgv = gauss_condition(mu, sigma, visible_nodes, visible_values)
        for rr, h in enumerate(hidden_nodes):
            x_imputed[i, h] = mu_hgv[rr]
    return x_imputed


def gauss_fit_em(X, max_iter=50, eps=1e-04):
    """
    Compute MLE of multivariate Gaussian given missing data using EM.
    """
    nr, nc = X.shape
    C = np.isnan(X) == False  # Identifying nan locations
    e = 0.0000001
    one_to_nc = np.arange(1, nc + 1, step=1)
    M = one_to_nc * (C == False) - 1  # Missing locations (-1 at locations where Nan is present in X)
    O = one_to_nc * C - 1  # Observed locations (-1 at locations where Nan is not present in X)

    # Generate initial Mu and Sigma
    Mu = np.nanmean(X, axis=0).reshape(-1, 1)
    Mu_new = Mu.copy()
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows,].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis=0))
    S_new = S.copy()

    # Start updating
    X_tilde = X.copy()
    no_conv = True
    iteration = 0

    while no_conv and iteration < max_iter:
        # E-step:
        EX = np.zeros((nc, 1))
        EXX = np.zeros((nc, nc))
        EXsum = np.zeros((nc, 1))
        EXXsum = np.zeros((nc, nc))
        Mu = Mu_new
        S = S_new
        for i in range(nr):
            if set(O[i,]) != set(one_to_nc - 1):  # Missing component exists

                m_indx = M[i,] != -1
                o_indx = O[i,] != -1
                M_i = M[i,][m_indx]  # Missing entries (u)
                O_i = O[i,][o_indx]  # Observed entries (o)

                Mui = Mu[np.ix_(M_i)] + (S[np.ix_(M_i, O_i)] @ np.linalg.pinv(S[np.ix_(O_i, O_i)] + e) @ (
                            X_tilde[i, np.ix_(O_i)].T - Mu[np.ix_(O_i)]))  # Expected stats for mean
                Vi = S[np.ix_(M_i, M_i)] - S[np.ix_(M_i, O_i)] @ np.linalg.inv(S[np.ix_(O_i, O_i)] + e) @ S[
                    np.ix_(M_i, O_i)].T  # Expected stats for sigma
                Mui = Mui.reshape(-1, 1)

                EX[np.ix_(O_i)] = X_tilde[i, np.ix_(O_i)].T
                EX[np.ix_(M_i)] = Mui

                EXX[np.ix_(M_i, M_i)] = EX[np.ix_(M_i)] * EX[np.ix_(M_i)].T + Vi
                EXX[np.ix_(O_i, O_i)] = EX[np.ix_(O_i)] * EX[np.ix_(O_i)].T
                EXX[np.ix_(O_i, M_i)] = EX[np.ix_(O_i)] * EX[np.ix_(M_i)].T
                EXX[np.ix_(M_i, O_i)] = EX[np.ix_(M_i)] * EX[np.ix_(O_i)].T

                EXsum = EXsum + EX
                EXXsum = EXXsum + EXX

        # M-step:
        Mu_new = EXsum / nr
        S_new = EXXsum / nr - Mu_new * Mu_new.T

        # Convergence condition:
        no_conv = np.linalg.norm(Mu - Mu_new) >= eps or np.linalg.norm(S - S_new, ord=2) >= eps
        iteration += 1

    return {'mu': Mu, 'Sigma': S, 'niter': iteration}