# Online training of a logistic regression model
# using Assumed Density Filtering (ADF).
# We compare the ADF result with MCMC sampling
# For further details, see the ADF paper:
#   * O. Zoeter, "Bayesian Generalized Linear Models in a Terabyte World,"
#     2007 5th International Symposium on Image and Signal Processing and Analysis, 2007,
#     pp. 435-440, doi: 10.1109/ISPA.2007.4383733.
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
from jax.scipy.stats import norm
from jax_cosmo.scipy import integrate
from functools import partial

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def sigmoid(z): return jnp.exp(z) / (1 + jnp.exp(z))


def log_sigmoid(z): return z - jnp.log1p(jnp.exp(z))


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
    log_term = y * log_sigmoid(eta) + (1 - y) * jnp.log1p(-sigmoid(eta))
    log_term = log_term + norm.logpdf(eta, mu, v)
    
    return jnp.exp(log_term)


def mt_func(eta, y, mu, v, Zt):
    log_term = y * log_sigmoid(eta) + (1 - y) * jnp.log1p(-sigmoid(eta))
    log_term = log_term + norm.logpdf(eta, mu, v)
    
    return eta * jnp.exp(log_term) / Zt


def vt_func(eta, y, mu, v, Zt):
    log_term = y * log_sigmoid(eta) + (1 - y) * jnp.log1p(-sigmoid(eta))
    log_term = log_term + norm.logpdf(eta, mu, v)
    
    return eta ** 2 * jnp.exp(log_term) / Zt


def adf_step(state, xs, prior_variance, lbound, ubound):
    mu_t, tau_t = state
    Phi_t, y_t = xs
    
    mu_t_cond = mu_t
    tau_t_cond = tau_t + prior_variance

    # prior predictive distribution
    m_t_cond = (Phi_t * mu_t_cond).sum()
    v_t_cond = (Phi_t ** 2 * tau_t_cond).sum()

    v_t_cond_sqrt = jnp.sqrt(v_t_cond)

    # Moment-matched Gaussian approximation elements
    Zt = integrate.romb(lambda eta: Zt_func(eta, y_t, m_t_cond, v_t_cond_sqrt), lbound, ubound)

    mt = integrate.romb(lambda eta: mt_func(eta, y_t, m_t_cond, v_t_cond_sqrt, Zt), lbound, ubound)

    vt = integrate.romb(lambda eta: vt_func(eta, y_t, m_t_cond, v_t_cond_sqrt, Zt), lbound, ubound)
    vt = vt - mt ** 2
    
    # Posterior estimation
    delta_m = mt - m_t_cond
    delta_v = vt - v_t_cond
    a = Phi_t * tau_t_cond / (Phi_t ** 2 * tau_t_cond).sum()
    mu_t = mu_t_cond + a * delta_m
    tau_t = tau_t_cond + a ** 2 * delta_v
    
    return (mu_t, tau_t), (mu_t, tau_t)


def plot_posterior_predictive(ax, X, Z, title, colors, cmap="RdBu_r"):
    ax.contourf(*Xspace, Z, cmap=cmap, alpha=0.7, levels=20)
    ax.scatter(*X.T, c=colors, edgecolors="gray", s=80)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()


# ** Generating training data **
key = random.PRNGKey(314)
n_datapoints, ndims = 50, 2
X, rows, cols = make_biclusters((n_datapoints, ndims), 2, noise=0.6,
                                random_state=3141, minval=-4, maxval=4)
y = rows[0] * 1.0

alpha = 1.0
init_noise = 1.0
Phi = jnp.c_[jnp.ones(n_datapoints)[:, None], X] # Design matrix
ndata, ndims = Phi.shape


# ** MCMC Sampling with BlackJAX **
sigma_mcmc = 0.8
w0 = random.multivariate_normal(key, jnp.zeros(ndims), jnp.eye(ndims) * init_noise)
energy = partial(E_base, Phi=Phi, y=y, alpha=alpha)
initial_state = mh.new_state(w0, energy)

mcmc_kernel = mh.kernel(energy, jnp.ones(ndims) * sigma_mcmc)
mcmc_kernel = jax.jit(mcmc_kernel)

n_samples = 5_000
burnin = 300
key_init = jax.random.PRNGKey(0)
states = inference_loop(key_init, mcmc_kernel, initial_state, n_samples)

chains = states.position[burnin:, :]
nsamp, _ = chains.shape

# ** Laplace approximation **
res = minimize(lambda x: energy(x) / len(y), w0, method="BFGS")
w_map = res.x
SN = jax.hessian(energy)(w_map)

# ** ADF inference **
prior_variance = 0.0
# Lower and upper bounds of integration. Ideally, we would like to
# integrate from -inf to inf, but we run into numerical issues.
lbound, ubound = -20, 20
mu_t = jnp.zeros(ndims)
tau_t = jnp.ones(ndims) * 1.0

init_state = (mu_t, tau_t)
xs = (Phi, y)

adf_loop = partial(adf_step, prior_variance=prior_variance, lbound=lbound, ubound=ubound)
(mu_t, tau_t), (mu_t_hist, tau_t_hist) = jax.lax.scan(adf_loop, init_state, xs)


# ** Estimating posterior predictive distribution **
xmin, ymin = X.min(axis=0) - 0.1
xmax, ymax = X.max(axis=0) + 0.1
step = 0.1
Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
_, nx, ny = Xspace.shape
Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])

# MCMC posterior predictive distribution
# maps m-dimensional features on an (i,j) grid times "s" m-dimensional samples to get
# "s" samples on an (i,j) grid of predictions
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
colors = ["black" if el else "white" for el in y]

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
w_batch_laplace_all = w_map
w_batch_laplace_std_all = jnp.sqrt(jnp.diag(SN))

w_batch_std_all = chains.std(axis=0)
timesteps = jnp.arange(n_datapoints)
lcolors = ["black", "tab:blue", "tab:red"]

elements = zip(mu_t_hist.T, tau_t_hist.T, w_batch_all, w_batch_std_all, w_batch_laplace_all, lcolors)
for i, (w_online, w_err_online, w_batch, w_batch_err, w_batch_laplace, c) in enumerate(elements):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.errorbar(timesteps, w_online, jnp.sqrt(w_err_online), c=c, label=f"$w_{i}$ online")
    ax.axhline(y=w_batch, c=lcolors[i], linestyle="--", label=f"$w_{i}$ batch (mcmc)")
    ax.axhline(y=w_batch_laplace, c=lcolors[i], linestyle="dotted",
               label=f"$w_{i}$ batch (Laplace)", linewidth=2)
    ax.fill_between(timesteps, w_batch - w_batch_err, w_batch + w_batch_err, color=c, alpha=0.1)
    ax.legend() #loc="lower left")
    ax.set_xlim(0, n_datapoints - 0.9)
    ax.set_xlabel("number samples")
    ax.set_ylabel(f"weights ({i})")
    plt.tight_layout()
    pml.savefig(f"adf-mcmc-online-hist-w{i}.pdf")

plt.show()
