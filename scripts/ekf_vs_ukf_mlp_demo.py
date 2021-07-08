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
from jax.random import PRNGKey, split, normal, multivariate_normal

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


def sample_observations(key, f, n_obs, xmin, xmax, x_noise=0.1, y_noise=3.0):
    key_x, key_y = split(key, 2)
    x_noise = normal(key_x, (n_obs,)) * x_noise
    y_noise = normal(key_y, (n_obs,)) * y_noise
    x = jnp.linspace(xmin, xmax, n_obs) + x_noise
    y = f(x) + y_noise
    X = np.c_[x, y]
    np.random.shuffle(X)
    x, y = jnp.array(X.T)
    return x, y

n_hidden = 6
def fwd_mlp(W, x):
    W1 = W[:n_hidden].reshape(n_hidden, 1)
    W2 = W[n_hidden: 2 * n_hidden].reshape(1, n_hidden)
    b1 = W[2 * n_hidden: 2 * n_hidden + n_hidden]
    b2 = W[-1]
    
    return W2 @ nn.tanh(W1 @ x + b1) + b2

def f(x): return x -10 * jnp.cos(x) * jnp.sin(x) + x ** 3
def fz(W): return W

# vectorised for multiple observations
fwd_mlp_obs = jax.vmap(fwd_mlp, in_axes=[None, 0])
# vectorised for multiple weights
fwd_mlp_weights = jax.vmap(fwd_mlp, in_axes=[1, None])
# vectorised for multiple observations and weights
fwd_mlp_obs_weights = jax.vmap(fwd_mlp_obs, in_axes=[0, None])


sigma = 0.1
n_in, n_out = 1, 1
n_params = n_in * n_hidden + n_hidden * n_out  + n_hidden + n_out

n_params = 1 * n_hidden + n_hidden * 1 + n_hidden + 1
W0 = jnp.zeros((n_params,))

key = PRNGKey(31415)
W0 = normal(key, (n_params,)) * sigma
Q = jnp.eye(n_params) * sigma ** 2
R = jnp.eye(1) * 0.9


n_obs = 200
key = PRNGKey(314)
xmin, xmax = -3, 3
x, y = sample_observations(key, f, n_obs, xmin, xmax)

ekf = ds.ExtendedKalmanFilter(fz, fwd_mlp, Q, R)
ekf_mu_hist, ekf_Sigma_hist = ekf.filter(W0, y[:, None], x[:, None])

step = -1
W_ekf, SW_ekf = ekf_mu_hist[step], ekf_Sigma_hist[step]
W_samples = multivariate_normal(key, W_ekf, SW_ekf, (100,))

xtest = jnp.linspace(x.min(), x.max(), 200)
sample_yhat_ekf = fwd_mlp_obs_weights(W_samples, xtest[:, None])


fig, ax = plt.subplots()
for sample in sample_yhat_ekf:
    ax.plot(xtest, sample, c="tab:gray", alpha=0.07)
ax.plot(xtest, sample_yhat_ekf.mean(axis=0))
ax.scatter(x, y, s=14, c="none", edgecolor="black", label="observations", alpha=0.5)
ax.set_xlim(x.min(), x.max())
ax.set_title("EKF + MLP")


alpha, beta, kappa = 1.0, 2.0, 3.0 - n_params
ukf = ds.UnscentedKalmanFilter(fz, lambda w, x: fwd_mlp_weights(w, x).T, Q, R, alpha, beta, kappa)
ukf_mu_hist, ukf_Sigma_hist = ukf.filter(W0, y, x[:, None])

W_ukf, SW_ukf = ukf_mu_hist[-1], ukf_Sigma_hist[-1]
W_samples = multivariate_normal(key, W_ukf, SW_ukf, (100,))

sample_yhat_ukf = fwd_mlp_obs_weights(W_samples, xtest[:, None])

fig, ax = plt.subplots()
for sample in sample_yhat_ukf:
    ax.plot(xtest, sample, c="tab:gray", alpha=0.07)
ax.plot(xtest, sample_yhat_ukf.mean(axis=0))
ax.scatter(x, y, s=14, c="none", edgecolor="black", label="observations", alpha=0.5)
ax.set_xlim(x.min(), x.max())
ax.set_title("UKF + MLP")

plt.show()