# Demo showcasing the training of an MLP with a single hidden layer using 
# Extended Kalman Filtering (EKF) and Unscented Kalman Filtering (UKF).
# In this demo, we consider the latent state to be the weights of an MLP.
#   The observed state at time t is the output of the MLP as influenced by the weights
#   at time t-1 and the covariate x[t].
#   The transition function between latent states is the identity function.
# For more information, see
#   * Neural Network Training Using Unscented and Extended Kalman Filter
#       https://juniperpublishers.com/raej/RAEJ.MS.ID.555568.php
#   * UKF-based training algorithm for feed-forward neural networks with
#     application to XOR classification problem
#       https://ieeexplore.ieee.org/document/6234549

# Author: Gerardo Durán-Martín (@gerdm)

import superimport

import jax
import numpy as np
import nlds_lib as ds
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from functools import partial
from jax.random import PRNGKey, permutation, split, normal, multivariate_normal


def mlp(W, x, n_hidden):
    """
    Multilayer Perceptron (MLP) with a single hidden unit and
    tanh activation function. The input-unit and the
    output-unit are both assumed to be unidimensional

    Parameters
    ----------
    W: array(2 * n_hidden + n_hidden + 1)
        Unravelled weights of the MLP
    x: array(1,)
        Singleton element to evaluate the MLP
    n_hidden: int
        Number of hidden units

    Returns
    -------
    * array(1,)
        Evaluation of MLP at the specified point
    """
    W1 = W[:n_hidden].reshape(n_hidden, 1)
    W2 = W[n_hidden: 2 * n_hidden].reshape(1, n_hidden)
    b1 = W[2 * n_hidden: 2 * n_hidden + n_hidden]
    b2 = W[-1]
    
    return W2 @ jnp.tanh(W1 @ x + b1) + b2


def sample_observations(key, f, n_obs, xmin, xmax, x_noise=0.1, y_noise=3.0):
    key_x, key_y, key_shuffle = split(key, 3)
    x_noise = normal(key_x, (n_obs,)) * x_noise
    y_noise = normal(key_y, (n_obs,)) * y_noise
    x = jnp.linspace(xmin, xmax, n_obs) + x_noise
    y = f(x) + y_noise
    X = np.c_[x, y]

    shuffled_ixs = permutation(key_shuffle, jnp.arange(n_obs))
    x, y = jnp.array(X[shuffled_ixs, :].T)
    return x, y


def plot_mlp_prediction(key, xobs, yobs, xtest, fw, w, Sw, ax, n_samples=100):
    W_samples = multivariate_normal(key, w, Sw, (n_samples,))
    sample_yhat = fw(W_samples, xtest[:, None])
    for sample in sample_yhat: # sample curves
        ax.plot(xtest, sample, c="tab:gray", alpha=0.07)
    ax.plot(xtest, sample_yhat.mean(axis=0)) # mean of posterior predictive
    ax.scatter(xobs, yobs, s=14, c="none", edgecolor="black", label="observations", alpha=0.5)
    ax.set_xlim(xobs.min(), xobs.max())


def plot_intermediate_steps(ax, fwd_func, intermediate_steps, xtest, mu_hist, Sigma_hist):
    """
    Plot the intermediate steps of the training process, all of them in the same plot
    but in different subplots.
    """
    for step, axi in zip(intermediate_steps, ax.flatten()):
        W_step, SW_step = mu_hist[step], Sigma_hist[step]
        x_step, y_step = x[:step], y[:step]
        plot_mlp_prediction(key, x_step, y_step, xtest, fwd_func, W_step, SW_step, axi)
        axi.set_title(f"step={step}")
    plt.tight_layout()

def plot_intermediate_steps_single(method, fwd_func, intermediate_steps, xtest, mu_hist, Sigma_hist):
    """
    Plot the intermediate steps of the training process, each one in a different plot.
    """
    for step in intermediate_steps:
        W_step, SW_step = mu_hist[step], Sigma_hist[step]
        x_step, y_step = x[:step], y[:step]
        _, axi = plt.subplots()
        plot_mlp_prediction(key, x_step, y_step, xtest, fwd_func, W_step, SW_step, axi)
        axi.set_title(f"step={step}")
        plt.tight_layout()
        pml.savefig(f'{method}-mlp-step-{step}.pdf')

if __name__ == "__main__":
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    def f(x): return x -10 * jnp.cos(x) * jnp.sin(x) + x ** 3
    def fz(W): return W

    # *** MLP configuration ***
    n_hidden = 6
    n_in, n_out = 1, 1
    n_params = (n_in + 1) * n_hidden + (n_hidden + 1) * n_out
    fwd_mlp = partial(mlp, n_hidden=n_hidden)
    # vectorised for multiple observations
    fwd_mlp_obs = jax.vmap(fwd_mlp, in_axes=[None, 0])
    # vectorised for multiple weights
    fwd_mlp_weights = jax.vmap(fwd_mlp, in_axes=[1, None])
    # vectorised for multiple observations and weights
    fwd_mlp_obs_weights = jax.vmap(fwd_mlp_obs, in_axes=[0, None])

    # *** Generating training and test data ***
    n_obs = 200
    key = PRNGKey(314)
    key_sample_obs, key_weights = split(key, 2)
    xmin, xmax = -3, 3
    sigma_y = 3.0
    x, y = sample_observations(key_sample_obs, f, n_obs, xmin, xmax, x_noise=0, y_noise=sigma_y)
    xtest = jnp.linspace(x.min(), x.max(), n_obs)

    # *** MLP Training with EKF ***

    W0 = normal(key_weights, (n_params,)) * 1 # initial random guess
    Q = jnp.eye(n_params) * 1e-4; # parameters do not change
    R = jnp.eye(1) * sigma_y**2; # observation noise is fixed
    Vinit = jnp.eye(n_params) * 100 # vague prior

    ekf = ds.ExtendedKalmanFilter(fz, fwd_mlp, Q, R)
    ekf_mu_hist, ekf_Sigma_hist = ekf.filter(W0, y[:, None], x[:, None], Vinit)

    # Plot final performance
    fig, ax = plt.subplots()
    step = -1
    W_ekf, SW_ekf = ekf_mu_hist[step], ekf_Sigma_hist[step]
    plot_mlp_prediction(key, x, y, xtest, fwd_mlp_obs_weights, W_ekf, SW_ekf, ax)
    ax.set_title("EKF + MLP")
    pml.savefig("ekf-mlp.pdf")

    # Plot intermediate performance
    intermediate_steps = [10, 20, 30, 40, 50, 60]
    fig, ax = plt.subplots(2, 2)
    plot_intermediate_steps(ax, fwd_mlp_obs_weights, intermediate_steps, xtest, ekf_mu_hist, ekf_Sigma_hist)
    plt.suptitle("EKF + MLP training")
    pml.savefig("ekf-mlp-training-steps.pdf")
    plot_intermediate_steps_single("ekf", fwd_mlp_obs_weights, intermediate_steps, xtest, ekf_mu_hist, ekf_Sigma_hist)

    use_ukf = True
    if use_ukf:
        Vinit = jnp.eye(n_params) * 5 # vague prior
        alpha, beta, kappa = 0.01, 2.0, 3.0 - n_params
        ukf = ds.UnscentedKalmanFilter(fz, lambda w, x: fwd_mlp_weights(w, x).T, Q, R, alpha, beta, kappa)
        ukf_mu_hist, ukf_Sigma_hist = ukf.filter(W0, y, x[:, None], Vinit)
        step = -1
        W_ukf, SW_ukf = ukf_mu_hist[step], ukf_Sigma_hist[step]

        fig, ax = plt.subplots()
        plot_mlp_prediction(key, x, y, xtest, fwd_mlp_obs_weights, W_ukf, SW_ukf, ax)
        ax.set_title("UKF + MLP")
        pml.savefig("ukf-mlp.pdf")

        fig, ax = plt.subplots(2, 2)
        plot_intermediate_steps(ax, fwd_mlp_obs_weights, intermediate_steps, xtest, ukf_mu_hist, ukf_Sigma_hist)
        plt.suptitle("UKF + MLP training")
        pml.savefig("ukf-mlp-training-steps.pdf")
        plot_intermediate_steps_single("ukf", fwd_mlp_obs_weights, intermediate_steps, xtest, ukf_mu_hist, ukf_Sigma_hist)

    plt.show()
