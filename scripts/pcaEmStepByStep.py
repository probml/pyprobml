import superimport

import numpy as np
from numpy.linalg import svd, eig
from scipy.linalg import orth
from matplotlib import pyplot as plt

import pyprobml_utils as pml
#from confidence_ellipse import confidence_ellipse
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# Source:
# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


np.warnings.filterwarnings('ignore')

np.random.seed(10)


n = 25
d = 2
mu0 = np.random.multivariate_normal(np.ravel(np.eye(1, d)), np.eye(d), 1)
Sigma = np.array([[1, -0.7], [-0.7, 1]])
X = np.random.multivariate_normal(np.ravel(mu0), Sigma, n)
k = 1

mu = np.mean(X, axis=0)
X = X - mu
X = X.T  # algorithm in book uses [d,n] dimensional X

[U, S, V] = svd(Sigma, 0)
Wtrue = V[:, :k]

[U, S, V] = svd(np.cov(X))
Wdata = V[:, :k]

W = np.random.rand(X.shape[0], k)

converged = 0
negmseNew = - np.inf
iterator = 0

while not converged:
    negmseOld = negmseNew

    Z = np.linalg.lstsq(np.dot(W.T, W), np.dot(W.T, X))
    Xrecon = np.dot(W, Z[0])

    Wortho = orth(W)
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    confidence_ellipse(X[0, :], X[1, :], axs, edgecolor='red')
    axs.plot(X[0, :], X[1, :], 'g*')
    axs.scatter(Xrecon[0, :], Xrecon[1, :], edgecolors='k', marker='o', facecolor="none", s=80)

    axs.plot(np.linspace(-3, 3, 20), float(Wortho[1]) / Wortho[0] * np.linspace(-3, 3, 20), 'c', linewidth=2)
    for i in range(len(X[0])):
        X_p = [X[0, i], Xrecon[0, i]]
        Y_p = [X[1, i], Xrecon[1, i]]
        axs.plot(X_p, Y_p, 'k')
    comp_mean = X.mean(axis=1)
    axs.scatter(comp_mean[0], comp_mean[1], marker='x', c='r', s=200)
    axs.set_title('E step {}'.format(iterator))
    pml.savefig(f'pcaEmStepByStepEstep{iterator}.pdf')


    W = np.dot(X, Z[0].T) / np.dot(Z[0], Z[0].T)
    negmseNew = -np.mean((np.ravel(Xrecon) - np.ravel(X) ** 2))
    converged = pml.convergence_test(negmseOld, negmseNew, 1e-2)

    Wortho = orth(W)
    Z = np.dot(X.T, Wortho)
    fig2, axs2 = plt.subplots(1, 1, figsize=(8, 8))
    [evals, evecs] = eig(np.dot(Z.T, Z) / n)
    perm = np.argsort(evals)
    evecs = evecs[:, perm]
    West = np.dot(W, evecs)
    Z = np.dot(X.T, West)
    Xrecon = np.dot(Z, West.T)
    confidence_ellipse(X[0, :], X[1, :], axs2, edgecolor='red')

    axs2.plot(X[0, :], X[1, :], 'g*')
    axs2.scatter(Xrecon[:, 0], Xrecon[:, 1], edgecolors='k', marker='o', facecolor="none", s=80)
    axs2.plot(np.linspace(-3, 3, 20), float(Wortho[1]) / Wortho[0] * np.linspace(-3, 3, 20), 'c', linewidth=2)
    for i in range(len(X[0])):
        X_p = [X[0, i], Xrecon[i, 0]]
        Y_p = [X[1, i], Xrecon[i, 1]]
        axs2.plot(X_p, Y_p, 'k')
    comp_mean = X.mean(axis=1)
    axs2.scatter(comp_mean[0], comp_mean[1], marker='x', c='r', s=200)
    axs2.set_title('M step {}'.format(iterator))
    pml.savefig(f'pcaEmStepByStepMstep{iterator}.pdf')

    #fig.savefig('../figures/pcaEmStepByStepEstep{}.pdf'.format(iterator))
    #fig2.savefig('../figures/pcaEmStepByStepMstep{}.pdf'.format(iterator))

    iterator = iterator + 1
