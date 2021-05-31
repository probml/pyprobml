# This script produces an illustration of Kalman filtering and smoothing
# Author: Gerardo Duran-Martin (@gerdm)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from matplotlib.patches import Ellipse
import linear_dynamical_systems_lib as lds
import pyprobml_utils as pml

def plot_tracking_values(observed, filtered, cov_hist, signal_label, ax):
    """
    observed: array(nsteps, 2)
        Array of observed values
    filtered: array(nsteps, state_size)
        Array of latent (hidden) values. We consider only the first
        two dimensions of the latent values
    cov_hist: array(nsteps, state_size, state_size)
        History of the retrieved (filtered) covariance matrices
    ax: matplotlib AxesSubplot
    """
    timesteps, _ = observed.shape
    ax.plot(observed[:, 0], observed[:, 1], marker="o", linewidth=0,
             markerfacecolor="none", markeredgewidth=2, markersize=8, label="observed", c="tab:green")
    ax.plot(*filtered[:, :2].T, label=signal_label, c="tab:red", marker="x", linewidth=2)
    for t in range(0, timesteps, 1):
        covn = cov_hist[t][:2, :2]
        # Eigenvalues correspond to entries in the diagonal
        # of a diagonal matrix
        σx, σy = jnp.sqrt(jnp.diag(covn[:2, :2]))
        # width and heighta are twice the distance from the mean to the 1.96σx (95% confidence)
        # perpendicular distance
        zscore = 1.96
        width = 2 * zscore * σx
        height = 2 * zscore * σy
        ellipse = Ellipse(filtered[t, :2], width, height, fill=False, linewidth=0.8)
        ax.add_patch(ellipse)
    ax.axis("equal")
    ax.legend()


def sample_filter_smooth(lds_model, key):
    """
    Sample from a linear dynamical system, apply the kalman filter
    (forward pass), and performs smoothing.

    Parameters
    ----------
    lds: LinearDynamicalSystem
        Instance of a linear dynamical system with known parameters

    Returns
    -------
    Dictionary with the following key, values
    * (z_hist) array(timesteps, state_size):
        Simulation of Latent states
    * (x_hist) array(timesteps, observation_size):
        Simulation of observed states
    * (μ_hist) array(timesteps, state_size):
        Filtered means μt
    * (Σ_hist) array(timesteps, state_size, state_size)
        Filtered covariances Σt
    * (μ_cond_hist) array(timesteps, state_size)
        Filtered conditional means μt|t-1
    * (Σ_cond_hist) array(timesteps, state_size, state_size)
        Filtered conditional covariances Σt|t-1
    * (μ_hist_smooth) array(timesteps, state_size):
        Smoothed means μt
    * (Σ_hist_smooth) array(timesteps, state_size, state_size)
        Smoothed covariances Σt
    """
    z_hist, x_hist = lds_model.sample(key)
    μ_hist, Σ_hist, μ_cond_hist, Σ_cond_hist = lds_model.kalman_filter(x_hist)
    μ_hist_smooth, Σ_hist_smooth = lds_model.kalman_smoother(μ_hist, Σ_hist, μ_cond_hist, Σ_cond_hist)

    return {
        "z_hist": z_hist,
        "x_hist": x_hist,
        "μ_hist": μ_hist,
        "Σ_hist": Σ_hist,
        "μ_cond_hist": μ_cond_hist,
        "Σ_cond_hist": Σ_cond_hist,
        "μ_hist_smooth": μ_hist_smooth,
        "Σ_hist_smooth": Σ_hist_smooth
    }


if __name__ == "__main__":
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    key = random.PRNGKey(314)
    timesteps = 15
    Δ = 1.0
    A = jnp.array([
        [1, 0, Δ, 0],
        [0, 1, 0, Δ],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    C = jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    state_size, _ = A.shape
    observation_size, _ = C.shape

    Q = jnp.eye(state_size) * 0.001
    R = jnp.eye(observation_size) * 1.0
    # Prior parameter distribution
    μ0 = jnp.array([8, 10, 1, 0])
    Σ0 = jnp.eye(state_size) * 1.0

    lds_instance = lds.LinearDynamicalSystem(A, C, Q, R, μ0, Σ0, timesteps)
    result = sample_filter_smooth(lds_instance, key)

    l2_filter = jnp.linalg.norm(result["z_hist"][:, :2] - result["μ_hist"][:, :2], 2)
    l2_smooth = jnp.linalg.norm(result["z_hist"][:, :2] - result["μ_hist_smooth"][:, :2], 2)

    print(f"L2-filter: {l2_filter:0.4f}")
    print(f"L2-smooth: {l2_smooth:0.4f}")

    fig, axs = plt.subplots()
    axs.plot(result["x_hist"][:, 0], result["x_hist"][:, 1], marker="o", linewidth=0,
         markerfacecolor="none", markeredgewidth=2, markersize=8, label="observed", c="tab:green")
    axs.plot(result["z_hist"][:, 0], result["z_hist"][:, 1], linewidth=2, label="truth", marker="s", markersize=8)
    axs.legend()
    axs.axis("equal")
    pml.savefig("kalman_tracking_truth.png")
    plt.show()

    fig, axs = plt.subplots()
    plot_tracking_values(result["x_hist"], result["μ_hist"], result["Σ_hist"], "filtered", axs)
    pml.savefig("kalman_tracking_filtered.png")
    plt.show()

    fig, axs = plt.subplots()
    plot_tracking_values(result["x_hist"], result["μ_hist_smooth"], result["Σ_hist_smooth"], "smoothed", axs)
    pml.savefig("kalman_tracking_smoothed.png")
    plt.show()
