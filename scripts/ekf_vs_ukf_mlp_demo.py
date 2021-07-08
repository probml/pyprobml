# Example of an training of a multilayered perceptron (MLP)
# using the Extended Kalman Filter (EKF) and the
# Unscented Kalman Filter
# Author: Gerardo Durán-Martín (@gerdm)

import jax
import numpy as np
import nlds_lib as ds
import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from numpy.random import shuffle, seed
from jax.random import PRNGKey, split, normal, multivariate_normal

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


def sample_observations(key, f, n_obs, xmin, xmax, x_noise=0.1, y_noise=3.0, shuffle_seed=None):
    key_x, key_y = split(key, 2)
    x_noise = normal(key_x, (n_obs,)) * x_noise
    y_noise = normal(key_y, (n_obs,)) * y_noise
    x = jnp.linspace(xmin, xmax, n_obs) + x_noise
    y = f(x) + y_noise
    X = np.c_[x, y]
    seed(shuffle_seed)
    shuffle(X)
    x, y = jnp.array(X.T)
    return x, y


def plot_mlp_prediction(key, xobs, yobs, xtest, fw, w, Sw, ax, n_samples=100):
    W_samples = multivariate_normal(key, w, Sw, (n_samples,))
    sample_yhat = fw(W_samples, xtest[:, None])

    for sample in sample_yhat:
        ax.plot(xtest, sample, c="tab:gray", alpha=0.07)
    ax.plot(xtest, sample_yhat.mean(axis=0))
    ax.scatter(xobs, yobs, s=14, c="none", edgecolor="black", label="observations", alpha=0.5)
    ax.set_xlim(xobs.min(), xobs.max())



def mlp(W, x, n_hidden):
    W1 = W[:n_hidden].reshape(n_hidden, 1)
    W2 = W[n_hidden: 2 * n_hidden].reshape(1, n_hidden)
    b1 = W[2 * n_hidden: 2 * n_hidden + n_hidden]
    b2 = W[-1]
    
    return W2 @ nn.tanh(W1 @ x + b1) + b2

def f(x): return x -10 * jnp.cos(x) * jnp.sin(x) + x ** 3
def fz(W): return W

# *** MLP configuration ***
n_hidden = 6
n_in, n_out = 1, 1
n_params = n_in * n_hidden + n_hidden * n_out  + n_hidden + n_out
fwd_mlp = partial(mlp, n_hidden=n_hidden)
# vectorised for multiple observations
fwd_mlp_obs = jax.vmap(fwd_mlp, in_axes=[None, 0])
# vectorised for multiple weights
fwd_mlp_weights = jax.vmap(fwd_mlp, in_axes=[1, None])
# vectorised for multiple observations and weights
fwd_mlp_obs_weights = jax.vmap(fwd_mlp_obs, in_axes=[0, None])

# *** Generating training and test data ***
n_obs = 200
shuffle_seed = 271
key = PRNGKey(314)
key_sample_obs, key_weights = split(key, 2)
xmin, xmax = -3, 3
x, y = sample_observations(key_sample_obs, f, n_obs, xmin, xmax, shuffle_seed=shuffle_seed)
xtest = jnp.linspace(x.min(), x.max(), n_obs)

# *** MLP Training with xKF ***
sigma = 0.1
W0 = normal(key_weights, (n_params,)) * sigma
Q = jnp.eye(n_params) * sigma ** 2
R = jnp.eye(1) * 0.9
alpha, beta, kappa = 1.0, 2.0, 3.0 - n_params
step = -1

ekf = ds.ExtendedKalmanFilter(fz, fwd_mlp, Q, R)
ekf_mu_hist, ekf_Sigma_hist = ekf.filter(W0, y[:, None], x[:, None])
W_ekf, SW_ekf = ekf_mu_hist[step], ekf_Sigma_hist[step]

ukf = ds.UnscentedKalmanFilter(fz, lambda w, x: fwd_mlp_weights(w, x).T, Q, R, alpha, beta, kappa)
ukf_mu_hist, ukf_Sigma_hist = ukf.filter(W0, y, x[:, None])
W_ukf, SW_ukf = ukf_mu_hist[step], ukf_Sigma_hist[step]


fig, ax = plt.subplots()
plot_mlp_prediction(key, x, y, xtest, fwd_mlp_obs_weights, W_ekf, SW_ekf, ax)
ax.set_title("EKF + MLP")

fig, ax = plt.subplots()
plot_mlp_prediction(key, x, y, xtest, fwd_mlp_obs_weights, W_ukf, SW_ukf, ax)
ax.set_title("UKF + MLP")

plt.show()