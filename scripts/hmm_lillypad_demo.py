# Example of an HMM with Gaussian emission in 2D
# For a matlab version, see https://github.com/probml/pmtk3/blob/master/demos/hmmLillypadDemo.m

# Author: Gerardo Durán-Martín (@gerdm), Aleyna Kara(@karalleyna)

import superimport

import logging
logging.getLogger('absl').setLevel(logging.CRITICAL)

import jax.numpy as jnp
from jax import vmap
from jax.random import PRNGKey

import numpy as np
import matplotlib.pyplot as plt

import distrax
from distrax import HMM
import tensorflow_probability as tfp

import pyprobml_utils as pml

def plot_2dhmm(hmm, samples_obs, samples_state, colors, ax, xmin, xmax, ymin, ymax, step=1e-2):
    """
    Plot the trajectory of a 2-dimensional HMM
    Parameters
    ----------
    hmm : HMM
        Hidden Markov Model
    samples_obs: numpy.ndarray(n_samples, 2)
        Observations
    samples_state: numpy.ndarray(n_samples, )
        Latent state of the system
    colors: list(int)
        List of colors for each latent state
    step: float
        Step size
    Returns
    -------
    * matplotlib.axes
    * colour of each latent state
    """
    obs_dist = hmm.obs_dist
    color_sample = [colors[i] for i in samples_state]

    xs = jnp.arange(xmin, xmax, step)
    ys = jnp.arange(ymin, ymax, step)

    v_prob = vmap(lambda x, y: obs_dist.prob(jnp.array([x, y])), in_axes=(None, 0))
    z = vmap(v_prob, in_axes=(0, None))(xs, ys)

    grid = np.mgrid[xmin:xmax:step, ymin:ymax:step]

    for k, color in enumerate(colors):
        ax.contour(*grid, z[:, :, k], levels=[1], colors=color, linewidths=3)
        ax.text(*(obs_dist.mean()[k] + 0.13), f"$k$={k + 1}", fontsize=13, horizontalalignment="right")

    ax.plot(*samples_obs.T, c="black", alpha=0.3, zorder=1)
    ax.scatter(*samples_obs.T, c=color_sample, s=30, zorder=2, alpha=0.8)

    return ax, color_sample

if __name__ == "__main__":
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    initial_probs = jnp.array([0.3, 0.2, 0.5])

    # transition matrix
    A = jnp.array([
        [0.3, 0.4, 0.3],
        [0.1, 0.6, 0.3],
        [0.2, 0.3, 0.5]
    ])

    S1 = jnp.array([
        [1.1, 0],
        [0, 0.3]
    ])

    S2 = jnp.array([
        [0.3, -0.5],
        [-0.5, 1.3]
    ])

    S3 = jnp.array([
        [0.8, 0.4],
        [0.4, 0.5]
    ])

    cov_collection = jnp.array([S1, S2, S3]) / 60
    mu_collection = jnp.array([
        [0.3, 0.3],
        [0.8, 0.5],
        [0.3, 0.8]
    ])

    hmm = HMM(trans_dist=distrax.Categorical(probs=A),
              init_dist=distrax.Categorical(probs=initial_probs),
              obs_dist=distrax.as_distribution(
                  tfp.substrates.jax.distributions.MultivariateNormalFullCovariance(loc=mu_collection,
                                                                                    covariance_matrix=cov_collection)))
    n_samples, seed = 50, 10
    samples_state, samples_obs = hmm.sample(seed=PRNGKey(seed), seq_len=n_samples)

    xmin, xmax = 0, 1
    ymin, ymax = 0, 1.2
    colors = ["tab:green", "tab:blue", "tab:red"]

    fig, ax = plt.subplots()
    _, color_sample = plot_2dhmm(hmm, samples_obs, samples_state, colors, ax, xmin, xmax, ymin, ymax)
    pml.savefig("hmm_lillypad_2d.pdf")

    fig, ax = plt.subplots()
    ax.step(range(n_samples), samples_state, where="post", c="black", linewidth=1, alpha=0.3)
    ax.scatter(range(n_samples), samples_state, c=color_sample, zorder=3)
    pml.savefig("hmm_lillypad_step.pdf")
    plt.show()