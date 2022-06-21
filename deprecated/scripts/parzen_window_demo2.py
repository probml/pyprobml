# Demonstrate a non-parametric (parzen) density estimator in 1D

# Author: Gerardo Durán Martín

import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

def K(u, axis=0): return np.all(np.abs(u) <= 1/2, axis=axis)

def p1(x, X, h):
    """
    KDE under a unit hypercube
    """
    N, D = X.shape
    xden, _ = x.shape
    
    u = ((x - X.T) / h).reshape(D, xden, N)
    ku = K(u).sum(axis=1) / (N * h ** D)
    return ku

def kdeg(x, X, h, return_components=False):
    """
    KDE under a gaussian kernel
    """
    N, D = X.shape
    nden, _ = x.shape
    
    Xhat = X.reshape(D, 1, N)
    xhat = x.reshape(D, nden, 1)
    u = xhat - Xhat
    u = norm(u, ord=2, axis=0) ** 2 / (2 * h ** 2) # (N, nden)
    px = np.exp(-u)
    if not return_components:
        px = px.sum(axis=1)

    px = px / (N * h * np.sqrt(2 * np.pi))
    return px


def main():
    data = np.array([-2.1, -1.3, -0.4, 1.9, 5.1, 6.2])[:, None]
    yvals = np.zeros_like(data)
    xv = np.linspace(-5, 10, 100)[:, None]

    fig, ax = plt.subplots(2, 2)
    # Uniform h=1
    ax[0,0].scatter(data, yvals, marker="x", c="tab:gray")
    ax[0,0].step(xv, p1(xv, data, 1), c="tab:blue", alpha=0.7)
    ax[0,0].set_title("unif, h=1.0")
    # Uniform h=2
    ax[0,1].scatter(data, yvals, marker="x", c="tab:gray")
    ax[0,1].step(xv, p1(xv, data, 2), c="tab:blue", alpha=0.7)
    ax[0,1].set_title("unif, h=2.0")

    # Gaussian h=1
    ax[1,0].scatter(data, yvals, marker="x", c="tab:gray", zorder=3)
    ax[1,0].plot(xv, kdeg(xv, data, 1), c="tab:blue", alpha=0.7, zorder=2)
    ax[1,0].plot(xv, kdeg(xv, data, 1, True), c="tab:red", alpha=0.7,
            linestyle="--", zorder=1, linewidth=1)
    ax[1,0].set_title("gauss, h=1.0")
    # Gaussian h=2
    ax[1,1].scatter(data, yvals, marker="x", c="tab:gray", zorder=3)
    ax[1,1].plot(xv, kdeg(xv, data, 2), c="tab:blue", alpha=0.7, zorder=2)
    ax[1,1].plot(xv, kdeg(xv, data, 2, True), c="tab:red", alpha=0.7,
                 linestyle="--", zorder=1, linewidth=1)
    ax[1,1].set_title("gauss, h=2.0")

    plt.tight_layout()
    plt.savefig("../figures/parzen_window2.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
