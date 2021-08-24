# Interpolate noise-free observations using a multivariate Gaussiab

import superimport

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import pyprobml_utils as pml

from matplotlib import colors as mcolors


def demo(priorVar, plot_num):
    np.random.seed(1)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    N = 10  # number of interior observed points
    D = 150  # number of points we evaluate function at
    xs = np.linspace(0, 1, D)
    allNdx = np.arange(0, D, 1)
    perm = np.random.permutation(D)
    obsNdx = perm[: N]
    obsNdx = np.concatenate((np.array([0]), obsNdx, np.array([D - 1])))
    Nobs = len(obsNdx)
    hidNdx = np.setdiff1d(allNdx, obsNdx)
    Nhid = len(hidNdx)
    xobs = np.random.randn(Nobs)
    obsNoiseVar = 1
    y = xobs + np.sqrt(obsNoiseVar) * np.random.randn(Nobs)
    L = (0.5 * scipy.sparse.diags([-1, 2, -1],
                                  [0, 1, 2], (D - 2, D))).toarray()

    Lambda = 1 / priorVar
    L = L * Lambda
    L1 = L[:, hidNdx]
    L2 = L[:, obsNdx]

    B11 = np.dot(np.transpose(L1), L1)
    B12 = np.dot(np.transpose(L1), L2)
    B21 = np.transpose(B12)

    mu = np.zeros(D)
    mu[hidNdx] = -np.dot(np.dot(np.linalg.inv(B11), B12), xobs)
    mu[obsNdx] = xobs
    inverseB11 = np.linalg.inv(B11)

    Sigma = np.zeros((D, D))
    # https://stackoverflow.com/questions/22927181/selecting-specific-rows-and-columns-from-numpy-array/22927889#22927889
    Sigma[hidNdx[:, None], hidNdx] = inverseB11

    plt.figure()
    plt.plot(obsNdx, xobs, 'bo', markersize=10)
    plt.plot(allNdx, mu, 'r-')

    S2 = np.diag(Sigma)
    upper = (mu + 2 * np.sqrt(S2))
    lower = (mu - 2 * np.sqrt(S2))
    plt.fill_between(allNdx, lower, upper, alpha=0.2)

    for i in range(0, 3):
        fs = np.random.multivariate_normal(mu, Sigma)
        plt.plot(allNdx, fs, 'k-', alpha=0.7)

    plt.title(f'prior variance {priorVar:0.2f}')
    pml.savefig(f'gaussian_interpolation_1d_{plot_num}.pdf')


priorVars = [0.01, 0.1]
for i, priorVar in enumerate(priorVars):
    demo(priorVar, i)
    plt.show()
