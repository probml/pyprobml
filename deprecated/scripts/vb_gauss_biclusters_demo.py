import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import random
from jax.scipy.stats import norm

from sklearn.datasets import make_biclusters
from functools import partial

import flax.linen as nn
import optax

import vb_gauss_cholesky as ffvb
import vb_gauss_lowrank as nagvac

import pyprobml_utils as pml


class LogReg(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=1, use_bias=False, kernel_init=nn.initializers.zeros)(x)


def plot_posterior_predictive(ax, X, Xspace, Zspace, title, colors, cmap="RdBu_r"):
    ax.contourf(*Xspace, Zspace, cmap=cmap, alpha=0.7, levels=20)
    ax.scatter(*X.T, c=colors, edgecolors="gray", s=80)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()


def loglikelihood_fn(params, Phi, y, predict_fn):
    an = predict_fn(params, Phi)
    log_an = nn.log_sigmoid(an)
    log_likelihood_term = y * log_an + (1 - y) * jnp.log(1 - nn.sigmoid(an))
    return log_likelihood_term.sum()


def logprior_fn(params, alpha=2.0):
    return -sum(jax.tree_leaves(jax.tree_map(lambda x: jnp.log(alpha * x.T @ x / 2).sum(), params)))


def logjoint_fn(params, data, predict_fn):
    return loglikelihood_fn(params, *data, predict_fn) + logprior_fn(params)


key = jax.random.PRNGKey(0)

## Data generating process
n_datapoints = 50
m = 2
noise = 0.6
bound = 4
X, rows, _ = make_biclusters((n_datapoints, m), 2,
                             noise=noise, random_state=3141,
                             minval=-bound, maxval=bound)

# whether datapoints belong to class 1
y = rows[0] * 1.0

Phi = jnp.c_[jnp.ones(n_datapoints)[:, None], X]
nfeatures = Phi.shape[-1]

# Model
model = LogReg()
init_key, key = jax.random.split(key)
variables = model.init(init_key, Phi)


# colors = ["black" if el else "white" for el in y]

# Predictive domain
xmin, ymin = X.min(axis=0) - 0.1
xmax, ymax = X.max(axis=0) + 0.1
step = 0.1
Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
_, nx, ny = Xspace.shape
Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])

### FFVB Approximation


partial_loglikelihood = partial(loglikelihood_fn,
                                predict_fn=lambda params, x: model.apply(params, x).squeeze())
partial_logjoint = partial(logjoint_fn,
                           predict_fn=lambda params, x: x @ params)

learning_rate = 1e-3
optimizer = optax.adam(learning_rate)

## Fixed Form Variational Bayes Approximation
(w_ffvb, lower_triangular), _ = ffvb.vb_gauss_chol(key, partial_loglikelihood,
                                                   logprior_fn, (Phi, y), optimizer, variables, niters=800)
w_ffvb = w_ffvb['params']['Dense_0']['kernel'].squeeze()
lower_triangular = lower_triangular['params']['Dense_0']['kernel']
cov_ffvb = lower_triangular @ lower_triangular.T

# Variational Bayes Low Rank Approximation
(w_lowrank, b, c), lower_bounds = nagvac.vb_gauss_lowrank(key, partial_logjoint, (Phi, y),
                                                          nfeatures, nsamples=20, niters=800, initial_std=0.1,
                                                          initial_scale=0.3,
                                                          initial_mean=-0.8 + 0.1 * random.normal(key, (nfeatures, 1)),
                                                          optimizer=optax.adafactor(1e-4))

w_lowrank = w_lowrank.squeeze()
cov_lowrank = b @ b.T + jnp.diag(c ** 2)

# *** Ploting surface predictive distribution ***
colors = ["black" if el else "white" for el in y]
key = random.PRNGKey(31415)
nsamples = 5000

# FFVB surface predictive distribution
ffvb_samples = random.multivariate_normal(key, w_ffvb, cov_ffvb, (nsamples,))
Z_ffvb = nn.sigmoid(jnp.einsum("mij,sm->sij", Phispace, ffvb_samples))
Z_ffvb = Z_ffvb.mean(axis=0)

# Variational Bayes Low Rank surface predictive distribution
lowrank_samples = random.multivariate_normal(key, w_lowrank, cov_lowrank, (nsamples,))

Z_lowrank = nn.sigmoid(jnp.einsum("mij,sm->sij", Phispace, lowrank_samples))
Z_lowrank = Z_lowrank.mean(axis=0)

fig_ffvb, ax = plt.subplots()
title = "FFVB  Predictive Distribution"
plot_posterior_predictive(ax, X, Xspace, Z_ffvb, title, colors)
pml.savefig('ffvb_predictive_distribution.pdf')
pml.savefig('ffvb_predictive_distribution.png')

fig_lowrank, ax = plt.subplots()
title = "NAGVAC Predictive Distribution"
plot_posterior_predictive(ax, X, Xspace, Z_lowrank, title, colors)
pml.savefig('nagvac_predictive_distribution.pdf')
pml.savefig('nagvac_predictive_distribution.png')

# *** Plotting posterior marginals of weights ***
for w, cov, method in [[w_ffvb, cov_ffvb, "ffvb"], [w_lowrank, cov_lowrank, "vb_lowrank"]]:
    for i in range(nfeatures):
        fig_weights_marginals, ax = plt.subplots()
        mean, std = w[i], jnp.sqrt(cov[i, i])
        x = jnp.linspace(mean - 4 * std, mean + 4 * std, 500)
        ax.plot(x, norm.pdf(x, mean, std), label=f"posterior ({method})", linestyle="dashdot")
        ax.legend()
        ax.set_title(f"Posterior marginals of weights ({i})")
        pml.savefig(f'{method}_weights_marginals_{i}.pdf')
        pml.savefig(f'{method}_weights_marginals_{i}.png')

print("FFVB weights")
print(w_ffvb, end="\n" * 2)

print("NAGVAC weights")
print(w_lowrank, end="\n" * 2)

plt.show()