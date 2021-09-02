
# Online training of a logistic regression model
# using Assumed Density Filtering (ADF).
# We compare the ADF result with MCMC sampling
# of the posterior distribution
# Dependencies:
#   !pip install git+https://github.com/blackjax-devs/blackjax.git
#   !pip install jax_cosmo

# Author: Gerardo Durán-Martín (@gerdm)

import superimport

import jax
import jax.numpy as jnp
import blackjax.rwmh as mh
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from sklearn.datasets import make_biclusters
from jax import random
from jax.scipy.optimize import minimize
from jax_cosmo.scipy import integrate
from functools import partial


def sigmoid(z): return jnp.exp(z) / (1 + jnp.exp(z))


def log_sigmoid(z): return z - jnp.log(1 + jnp.exp(z))


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def E_base(w, Phi, y, alpha):
    """
    Base function containing the Energy of a logistic
    regression with
    """
    an = Phi @ w
    log_an = log_sigmoid(an)
    log_likelihood_term = y * log_an + (1 - y) * jnp.log(1 - sigmoid(an))
    prior_term = alpha * w @ w / 2

    return prior_term - log_likelihood_term.sum()


def Zt_func(eta, y, mu, v):
    log_term = y * log_sigmoid(eta) + (1 - y) * jnp.log(1 - sigmoid(eta))
    log_term = log_term - (eta - mu) ** 2 / (2 * v ** 2)

    return jnp.exp(log_term) / jnp.sqrt(2 * jnp.pi * v ** 2)


def mt_func(eta, y, mu, v):
    log_term = y * log_sigmoid(eta) + (1 - y) * jnp.log(1 - sigmoid(eta))
    log_term = log_term - (eta - mu) ** 2 / (2 * v ** 2)

    return eta * jnp.exp(log_term) / jnp.sqrt(2 * jnp.pi * v ** 2)


def vt_func(eta, y, mu, v):
    log_term = y * log_sigmoid(eta) + (1 - y) * jnp.log(1 - sigmoid(eta))
    log_term = log_term - (eta - mu) ** 2 / (2 * v ** 2)

    return eta ** 2 * jnp.exp(log_term) / jnp.sqrt(2 * jnp.pi * v ** 2)


def adf_step(state, xs, q, lbound, ubound):
    mu_t, tau_t = state
    Phi_t, y_t = xs

    mu_t_cond = mu_t
    tau_t_cond = tau_t + q

    # prior predictive distribution
    m_t_cond = (Phi_t * mu_t_cond).sum()
    v_t_cond = (Phi_t ** 2 * tau_t_cond).sum()

    v_t_cond_sqrt = jnp.sqrt(v_t_cond)

    # Moment-matched Gaussian approximation elements
    Zt = integrate.romb(lambda eta: Zt_func(eta, y_t, m_t_cond, v_t_cond_sqrt), lbound, ubound)

    mt = integrate.romb(lambda eta: mt_func(eta, y_t, m_t_cond, v_t_cond_sqrt), lbound, ubound)
    mt = mt / Zt

    vt = integrate.romb(lambda eta: vt_func(eta, y_t, m_t_cond, v_t_cond_sqrt), lbound, ubound)
    vt = vt / Zt - mt ** 2

    # Posterior estimation
    delta_m = mt - m_t_cond
    delta_v = vt - v_t_cond
    a = Phi_t * tau_t_cond / jnp.power(Phi_t * tau_t_cond, 2).sum()
    mu_t = mu_t_cond + a * delta_m
    tau_t = tau_t_cond + a ** 2 * delta_v

    return (mu_t, tau_t), (mu_t, tau_t)


def plot_posterior_predictive(ax, X, Z, title, colors, cmap="RdBu_r"):
    ax.contourf(*Xspace, Z, cmap=cmap, alpha=0.5, levels=20)
    ax.scatter(*X.T, c=colors)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()


# ** Generating training data **
key = random.PRNGKey(314)
n_datapoints, m = 20, 2
X, rows, cols = make_biclusters((n_datapoints, m), 2, noise=0.6,
                                random_state=314, minval=-3, maxval=3)
y = rows[0] * 1.0

alpha = 1.0
init_noise = 1.0
Phi = jnp.c_[jnp.ones(n_datapoints)[:, None], X]
N, M = Phi.shape

# ** MCMC Sampling with BlackJAX **
sigma_mcmc = 0.8
w0 = random.multivariate_normal(key, jnp.zeros(M), jnp.eye(M) * init_noise)
E = partial(E_base, Phi=Phi, y=y, alpha=alpha)
initial_state = mh.new_state(w0, E)

mcmc_kernel = mh.kernel(E, jnp.ones(M) * sigma_mcmc)
mcmc_kernel = jax.jit(mcmc_kernel)

n_samples = 5_000
burnin = 300
key_init = jax.random.PRNGKey(0)
states = inference_loop(key_init, mcmc_kernel, initial_state, n_samples)

chains = states.position[burnin:, :]
nsamp, _ = chains.shape

# ** Laplace approximation **
res = minimize(lambda x: E(x) / len(y), w0, method="BFGS")
w_map = res.x
SN = jax.hessian(E)(w_map)

# ** ADF inference **
q = 0.14
lbound, ubound = -10, 10
mu_t = jnp.zeros(M)
tau_t = jnp.ones(M) * q

init_state = (mu_t, tau_t)
xs = (Phi, y)

adf_loop = partial(adf_step, q=q, lbound=lbound, ubound=ubound)
(mu_t, tau_t), (mu_t_hist, tau_t_hist) = jax.lax.scan(adf_loop, init_state, xs)

# ** Estimating posterior predictive distribution **
xmin, ymin = X.min(axis=0) - 0.1
xmax, ymax = X.max(axis=0) + 0.1
step = 0.1
Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
_, nx, ny = Xspace.shape
Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])

# MCMC posterior predictive distribution
Z_mcmc = sigmoid(jnp.einsum("mij,sm->sij", Phispace, chains))
Z_mcmc = Z_mcmc.mean(axis=0)
# Laplace posterior predictive distribution
key = random.PRNGKey(314)
laplace_samples = random.multivariate_normal(key, w_map, SN, (n_samples,))
Z_laplace = sigmoid(jnp.einsum("mij,sm->sij", Phispace, laplace_samples))
Z_laplace = Z_laplace.mean(axis=0)
# ADF posterior predictive distribution
adf_samples = random.multivariate_normal(key, mu_t, jnp.diag(tau_t), (n_samples,))
Z_adf = sigmoid(jnp.einsum("mij,sm->sij", Phispace, adf_samples))
Z_adf = Z_adf.mean(axis=0)

# ** Plotting predictive distribution **
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
colors = ["tab:red" if el else "tab:blue" for el in y]

fig, ax = plt.subplots()
title = "(MCMC) Predictive distribution"
plot_posterior_predictive(ax, X, Z_mcmc, title, colors)
pml.savefig("mcmc-logreg-predictive-surface.pdf")

fig, ax = plt.subplots()
title = "(Laplace) Predictive distribution"
plot_posterior_predictive(ax, X, Z_adf, title, colors)
pml.savefig("laplace-logreg-predictive-surface.pdf")

fig, ax = plt.subplots()
title = "(ADF) Predictive distribution"
plot_posterior_predictive(ax, X, Z_adf, title, colors)
pml.savefig("adf-logreg-predictive-surface.pdf")

# ** Plotting training history **
w_batch_all = chains.mean(axis=0)
w_batch_std_all = chains.std(axis=0)
timesteps = jnp.arange(n_datapoints)
lcolors = ["black", "tab:blue", "tab:red"]

fig, ax = plt.subplots(figsize=(6, 3))
elements = zip(mu_t_hist.T, tau_t_hist.T, w_batch_all, w_batch_std_all, lcolors)
for i, (w_online, w_err_online, w_batch, w_batch_err, c) in enumerate(elements):
    ax.errorbar(timesteps, w_online, jnp.sqrt(w_err_online), c=c, label=f"$w_{i}$ online")
    ax.axhline(y=w_batch, c=lcolors[i], linestyle="--", label=f"$w_{i}$ batch (mcmc)")
    ax.fill_between(timesteps, w_batch - w_batch_err, w_batch + w_batch_err, color=c, alpha=0.1)
ax.legend(bbox_to_anchor=(1.05, 1))
ax.set_xlim(0, n_datapoints - 0.9)
ax.set_xlabel("number samples")
ax.set_ylabel("weights")
plt.tight_layout()
pml.savefig("adf-mcmc-online-hist.pdf")

plt.show()