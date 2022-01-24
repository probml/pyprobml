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


class LogReg(nn.Module):
    nfeatures: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=1, use_bias=False, kernel_init=nn.initializers.zeros)(x)


def plot_posterior_predictive(ax, X, Xspace, Zspace, title, colors, cmap="RdBu_r"):
    ax.contourf(*Xspace, Zspace, cmap=cmap, alpha=0.7, levels=20)
    ax.scatter(*X.T, c=colors, edgecolors="gray", s=80)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()


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
    return sum(jax.tree_leaves(jax.tree_map(lambda x: jnp.log(alpha * x.T @ x / 2).sum(), params)))


def main():
    key = jax.random.PRNGKey(0)

    ## Data generating process
    n_datapoints = 50
    m = 2
    X, rows, _ = make_biclusters((n_datapoints, m), 2,
                                 noise=0.6, random_state=3141,
                                 minval=-4, maxval=4)

    # whether datapoints belong to class 1
    y = rows[0] * 1.0

    Phi = jnp.c_[jnp.ones(n_datapoints)[:, None], X]
    nfeatures = Phi.shape[-1]

    # colors = ["black" if el else "white" for el in y]

    # Predictive domain
    xmin, ymin = X.min(axis=0) - 0.1
    xmax, ymax = X.max(axis=0) + 0.1
    step = 0.1
    Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
    _, nx, ny = Xspace.shape
    Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])

    ### FFVB Approximation
    model = LogReg(nfeatures)
    init_key, key = jax.random.split(key)
    variables = model.init(init_key, Phi)

    partial_loglikelihood = partial(loglikelihood_fn, predict_fn=lambda params, x: model.apply(params, x).squeeze())

    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)

    ## Fixed Form Variational Bayes Approximation
    (w_ffvb, cov_ffvb), _ = ffvb.vb_gauss_chol(key, partial_loglikelihood,
                                                 logprior_fn, (Phi, y), optimizer, variables, niters=800)
    w_ffvb = w_ffvb['params']['Dense_0']['kernel'].squeeze()
    cov_ffvb = cov_ffvb['params']['Dense_0']['kernel']
    cov_ffvb = cov_ffvb @ cov_ffvb.T

    ### *** Ploting surface predictive distribution ***
    colors = ["black" if el else "white" for el in y]
    dict_figures = {}
    key = random.PRNGKey(31415)
    nsamples = 5000

    # FFVB surface predictive distribution
    ffvb_samples = random.multivariate_normal(key, w_ffvb, cov_ffvb, (nsamples,))
    Z_ffvb = nn.sigmoid(jnp.einsum("mij,sm->sij", Phispace, ffvb_samples))
    Z_ffvb =    Z_ffvb.mean(axis=0)

    fig_ffvb, ax = plt.subplots()
    title = "FFVB  Predictive Distribution"
    plot_posterior_predictive(ax, X, Xspace, Z_ffvb, title, colors)
    plt.savefig('../figures/ffvb_predictive_distribution.pdf', dpi=300)


    # *** Plotting posterior marginals of weights ***
    for i in range(nfeatures):
        fig_weights_marginals, ax = plt.subplots()
        mean_ffvb, std_ffvb = w_ffvb[i], jnp.sqrt(cov_ffvb[i, i])
        x = jnp.linspace(mean_ffvb - 4 * std_ffvb, mean_ffvb + 4 * std_ffvb, 500)
        ax.plot(x, norm.pdf(x, mean_ffvb, std_ffvb), label="posterior (FFVB)", linestyle="dashdot")
        ax.legend()
        ax.set_title(f"Posterior marginals of weights ({i})")
        # dict_figures[f"weights_marginals_w{i}"] = fig_weights_marginals
        plt.savefig(f'../figures/ffvb_weights_marginals_{i}.pdf', dpi=300)

    print("FFVB weights")
    print(w_ffvb, end="\n" * 2)
    return dict_figures


if __name__ == "__main__":
    figs = main()
