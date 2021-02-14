# Visualize fitting a mixture of Gaussians to the old faithful dataset
# reproduce Bishop fig 9.8

#  Author: Gerardo Durán Martín

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap
from numpy.random import randn, seed


def create_colormap():
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(31/256, 214/256, N)
    vals[:, 1] = np.linspace(119/256, 39/256, N)
    vals[:, 2] = np.linspace(180/256, 40/256, N)
    cmap = ListedColormap(vals)
    return cmap



def compute_responsibilities(k, pi, mu, sigma):
    Ns = [multivariate_normal(mean=mu_i, cov=Sigma_i) for mu_i, Sigma_i in zip(mu, sigma)]
    def respons(x):
        elements = [pi_i * Ni.pdf(x) for pi_i, Ni in zip(pi, Ns)]
        return elements[k] / np.sum(elements, axis=0)
    return respons


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


def e_step(pi, mu, Sigma):
    responsibilities = []
    for i, _ in enumerate(mu):
        resp_k = compute_responsibilities(i, pi, mu, Sigma)
        responsibilities.append(resp_k)
    return responsibilities


def m_step(X, responsibilities):
    N, M = X.shape
    pi, mu, Sigma = [], [], []
    for resp_k in responsibilities:
        resp_k = resp_k(X)
        Nk = resp_k.sum()
        # mu_k
        mu_k = (resp_k[:, np.newaxis] * X).sum(axis=0) / Nk
        # Sigma_k
        dk = (X - mu_k)
        Sigma_k = resp_k[:, np.newaxis, np.newaxis] * np.einsum("ij, ik->ikj", dk, dk)
        Sigma_k = Sigma_k.sum(axis=0) / Nk
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


def apply_em(X, pi, mu, Sigma, threshold=1e-5):
    r = compute_responsibilities(0, pi, mu, Sigma)(X)
    log_likelihood = gmm_log_likelihood(X, pi, mu, Sigma)
    hist_log_likelihood = [log_likelihood]
    hist_coeffs = [(pi, mu, Sigma)]
    hist_responsibilities = [r]

    while True:
        responsibilities = e_step(pi, mu, Sigma)
        pi, mu, Sigma = m_step(X, responsibilities)
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

        
def main():
    cmap = create_colormap()
    seed(314)
    X = np.loadtxt("../data/faithful.txt")
    # Normalise data
    X = (X - X.mean(axis=0)) / (X.std(axis=0))
    mu1 = np.array([-1.5, 1.5])
    mu2 = np.array([1.5, -1.5])

    # Initial configuration
    Sigma1 = np.identity(2) * 0.1
    Sigma2 = np.identity(2) * 0.1
    pi = [0.5, 0.5]
    mu = [mu1, mu2]
    Sigma = [Sigma1, Sigma2]

    res = apply_em(X, pi, mu, Sigma)

    # Create grid-plot
    hist_index = [0, 10, 25, 30, 35, 40]
    fig, ax = plt.subplots(2, 3)
    ax = ax.ravel()
    for ix, axi in zip(hist_index, ax):
        pi, mu, Sigma = res["coeffs"][ix]
        r = res["rvals"][ix]
        if ix == 0:
            r = np.ones_like(r)

        colors = cmap if ix > 0 else "Dark2"
        plot_mixtures(X, mu, pi, Sigma, r, cmap=colors, ax=axi)
        axi.set_title("Iteration {ix}".format(ix=ix))
        
    plt.tight_layout()
    plt.savefig('../figures/gmm_faithful.pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
