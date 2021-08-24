'''
Visualize fitting a mixture of Gaussians by gradient descent to the old faithful dataset
reproduce Bishop fig 9.8
Author: Aleyna Kara(@karalleyna)
'''

# 64-bit precision is needed to attain the same results when scipy.stats.multivariate_normal is used.

#!pip install distrax

import superimport

from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyprobml_utils as pml
import requests
from io import BytesIO

from mix_gauss_lib import GMM
from mix_gauss_em_faithful import create_colormap

# 64-bit precision is needed to attain the same results when scipy.stats.multivariate_normal is used.
from jax.config import config
config.update("jax_enable_x64", True)

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

    num_epochs = 2000
    history = gmm.fit_sgd(jnp.array(observations), batch_size=observations.shape[0], num_epochs=num_epochs)
    ll_hist, mix_dist_probs_hist, comp_dist_loc_hist, comp_dist_cov_hist, responsibility_hist = history

    # Create grid-plot
    hist_index = [0, 10, 125, 320, 1450, 1999]
    fig, ax = plt.subplots(2, 3)
    ax = ax.ravel()

    for idx, axi in zip(hist_index, ax):
        means = comp_dist_loc_hist[idx]
        covariances = comp_dist_cov_hist[idx]
        responsibility = responsibility_hist[idx]

        if idx == 0:
            responsibility = jnp.ones_like(responsibility)

        color_map = cmap if idx > 0 else "Dark2"
        gmm.plot(observations, means, covariances, responsibility[:, 0], cmap=color_map, colors=colors, ax=axi)
        axi.set_title(f"Iteration {idx}")

    plt.tight_layout()
    pml.savefig('gmm_faithful.pdf')
    plt.show()

if __name__ == "__main__":
    main()

