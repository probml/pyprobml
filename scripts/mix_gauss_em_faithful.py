#!pip install distrax

'''
Visualize fitting a mixture of Gaussians by em algorithm to the old faithful dataset
reproduce Bishop fig 9.8
Author: Gerardo Durán-Martín, Aleyna Kara(@karalleyna)
'''
import superimport

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyprobml_utils as pml

from mix_gauss_lib import GMM
from matplotlib.colors import ListedColormap

import requests
from io import BytesIO

# 64-bit precision is needed to attain the same results when scipy.stats.multivariate_normal is used.
from jax.config import config
config.update("jax_enable_x64", True)

def create_colormap():
    # Creates color map
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(31 / 256, 214 / 256, N)
    vals[:, 1] = np.linspace(119 / 256, 39 / 256, N)
    vals[:, 2] = np.linspace(180 / 256, 40 / 256, N)
    cmap = ListedColormap(vals)
    return cmap

def main():
    cmap = create_colormap()
    colors = ["tab:red", "tab:blue"]

    url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/faithful.txt'
    response = requests.get(url)
    rawdata = BytesIO(response.content)
    observations = np.loadtxt(rawdata)
    # Normalize data
    observations = (observations - observations.mean(axis=0)) / (observations.std(axis=0))
    # Initial configuration

    mixing_coeffs = jnp.array([0.5, 0.5])

    means = jnp.vstack([jnp.array([-1.5, 1.5]),
                       jnp.array([1.5, -1.5])])

    covariances = jnp.array([jnp.eye(2) * 0.1,
                            jnp.eye(2) * 0.1])

    gmm = GMM(mixing_coeffs, means, covariances)
    num_of_iters = 50
    history = gmm.fit_em(observations, num_of_iters=num_of_iters)
    ll_hist, mix_dist_probs_hist, comp_dist_loc_hist, comp_dist_cov_hist, responsibility_hist = history

    # Create grid-plot
    hist_index = [0, 10, 25, 30, 35, 40]
    fig, ax = plt.subplots(2, 3)
    ax = ax.ravel()

    for idx, axi in zip(hist_index, ax):
        means = comp_dist_loc_hist[idx]
        covariances = comp_dist_cov_hist[idx]
        responsibility = responsibility_hist[idx]

        if idx == 0:
            responsibility = np.ones_like(responsibility)

        color_map = cmap if idx > 0 else "Dark2"
        gmm.plot(observations, means, covariances, responsibility, cmap=color_map, colors=colors, ax=axi)
        axi.set_title(f"Iteration {idx}")

    plt.tight_layout()
    pml.savefig('gmm_faithful.pdf')
    plt.show()

if __name__ == "__main__":
    main()

