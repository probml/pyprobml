# Example of an HMM with Gaussian emission in 2D
# For a matlab version, see https://github.com/probml/pmtk3/blob/master/demos/hmmLillypadDemo.m

# Author: Gerardo Durán-Martín (@gerdm)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pyprobml_utils as pml


def sample_hmm(n_samples, initial_probs, mu_collection, cov_collection):
    """
    Sample from a 2-dimensional HMM

    Parameters
    ----------
    n_samples: int
        Number of samples to be generated
    initial_probs: numpy.ndarray(n_states, )
        Initial probabilities for each state
    mu_collection: np.array(n_states, dim)
        Collection of means for each state
    cov_collection: np.array(n_states, dim, dim)
        Collection of covariances for each state
    
    Returns
    -------
    * numpy.ndarray(n_samples, d)
        Observations
    * numpy.ndarray(n_samples, )
        Latent states of the system
    """
    samples_obs = np.zeros((n_samples, 2))
    samples_state = np.zeros((n_samples,), dtype=int)
    zi = np.random.choice(z, p=initial_probs)
    N = multivariate_normal(mu_collection[zi], cov_collection[zi])
    samples_obs[0] = N.rvs()
    samples_state[0] = zi

    for i in range(1, n_samples):
        zi = np.random.choice(z, p=A[zi])
        N = multivariate_normal(mu_collection[zi], cov_collection[zi])
        samples_obs[i] = N.rvs()
        samples_state[i] = zi
    return samples_obs, samples_state


def plot_2dhmm(samples_obs, samples_state, colors, ax, xmin, xmax, ymin, ymax):
    """
    Plot the trajectory of a 2-dimensional HMM 

    Parameters
    ----------
    samples_obs: numpy.ndarray(n_samples, 2)
        Observations
    samples_state: numpy.ndarray(n_samples, )
        Latent state of the system
    colors: list(int)
        List of colors for each latent state

    Returns
    -------
    * matplotlib.axes
    * colour of each latent state
    """
    color_sample = [colors[i] for i in samples_state]
    Xgrid = np.mgrid[xmin:xmax:0.01, ymin:ymax:0.01]

    for k, (mu, S, color) in enumerate(zip(mu_collection, cov_collection, colors)):
        N = multivariate_normal(mean=mu, cov=S)
        Z = np.apply_along_axis(N.pdf, 0, Xgrid)
        ax.contour(*Xgrid, Z, levels=[1], colors=color, linewidths=3)
        ax.text(*(mu + 0.13), f"$k$={k + 1}", fontsize=13, horizontalalignment="right")
    ax.plot(*samples_obs.T, c="black", alpha=0.3, zorder=1)
    ax.scatter(*samples_obs.T, c=color_sample, s=30, zorder=2, alpha=0.8)
    return ax, color_sample


if __name__ == "__main__":
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    initial_probs = np.array([0.3, 0.2, 0.5])

    # transition matrix
    A = np.array([
        [0.3, 0.4, 0.3],
        [0.1, 0.6, 0.3],
        [0.2, 0.3, 0.5]
    ])

    S1 = np.array([
        [1.1, 0],
        [0, 0.3]
    ])

    S2 = np.array([
        [0.3, -0.5],
        [-0.5, 1.3]
    ])

    S3 = np.array([
        [0.8, 0.4],
        [0.4, 0.5]
    ])

    cov_collection = np.array([S1, S2, S3]) / 60
    mu_collection = np.array([
        [0.3, 0.3],
        [0.8, 0.5],
        [0.3, 0.8]
    ])

    np.random.seed(314)
    n_samples = 50
    z = np.array([0, 1, 2])
    colors = ["tab:green", "tab:blue", "tab:red"]

    xmin, xmax = 0, 1
    ymin, ymax = 0, 1.2
    samples_obs, samples_state = sample_hmm(n_samples, initial_probs, mu_collection, cov_collection)

    fig, ax = plt.subplots()
    _, color_sample = plot_2dhmm(samples_obs, samples_state, colors, ax, xmin, xmax, ymin, ymax)
    pml.savefig("hmm_lillypad_2d.pdf")

    fig, ax = plt.subplots()
    ax.step(range(n_samples), samples_state, where="post", c="black", linewidth=1, alpha=0.3)
    ax.scatter(range(n_samples), samples_state, c=color_sample, zorder=3)
    pml.savefig("hmm_lillypad_step.pdf")
    plt.show()
