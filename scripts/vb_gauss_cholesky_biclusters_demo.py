# !pip install git+git://github.com/probml/jsl

from itertools import chain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import blackjax.rmh as rmh

from jax import random
from jax.scipy.optimize import minimize
from jax.scipy.stats import norm

from sklearn.datasets import make_biclusters
from functools import partial

import flax.linen as nn
import optax

from jsl.nlds.extended_kalman_filter import ExtendedKalmanFilter
import vb_gauss_cholesky as ffvb


class LogReg(nn.Module):
    nfeatures: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=1, use_bias=False, kernel_init=nn.initializers.zeros)(x)


def fz(x): return x


def fx(w, x): return nn.sigmoid(w[None, :] @ x)


def Rt(w, x): return (nn.sigmoid(w @ x) * (1 - nn.sigmoid(w @ x)))[None, None]


def plot_posterior_predictive(ax, X, Xspace, Zspace, title, colors, cmap="RdBu_r"):
    ax.contourf(*Xspace, Zspace, cmap=cmap, alpha=0.7, levels=20)
    ax.scatter(*X.T, c=colors, edgecolors="gray", s=80)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()


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
    regression with. Energy log-joint
    """

    def predict_fn(w, Phi):
        return Phi @ w

    loglikelihood = loglikelihood_fn(w, Phi, y, predict_fn)
    logprior = jnp.exp(logprior_fn(w, alpha=alpha))
    return -logprior + loglikelihood



def mcmc_logistic_posterior_sample(key, Phi, y, alpha=1.0, init_noise=1.0,
                                   n_samples=5_000, burnin=300, sigma_mcmc=0.8):
    """
    Sample from the posterior distribution of the weights
    of a 2d binary logistic regression model p(y=1|x,w) = sigmoid(w'x),
    using the Metropolis-Hastings algorithm.
    """
    _, ndims = Phi.shape
    key, key_init = random.split(key)
    w0 = random.multivariate_normal(key, jnp.zeros(ndims), jnp.eye(ndims) * init_noise)
    energy = partial(E_base, Phi=Phi, y=y, alpha=alpha)
    initial_state = rmh.new_state(w0, energy)

    mcmc_kernel = rmh.kernel(energy, sigma=jnp.ones(ndims) * sigma_mcmc)
    mcmc_kernel = jax.jit(mcmc_kernel)

    states = inference_loop(key_init, mcmc_kernel, initial_state, n_samples)
    chains = states.position[burnin:, :]
    return chains


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

    partial_loglikelihood = partial(loglikelihood_fn,
                                    predict_fn=lambda params, x: model.apply(params, x).squeeze())

    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)

    ## Fixed Form Variational Bayes Approximation
    (w_ffvb, cov_ffvb), _ = ffvb.vb_gauss_chol(key, (Phi, y), optimizer, variables, model.apply,
                                               loglikelihood_fn=partial_loglikelihood,
                                               logprior_fn=logprior_fn)

    w_ffvb = w_ffvb['params']['Dense_0']['kernel'].squeeze()
    cov_ffvb = cov_ffvb['params']['Dense_0']['kernel']

    ### EEKF Approximation
    mu_t = jnp.zeros(nfeatures)
    Pt = jnp.eye(nfeatures) * 0.0
    P0 = jnp.eye(nfeatures) * 2.0

    model = ExtendedKalmanFilter(fz, fx, Pt, Rt)
    (w_eekf, P_eekf), eekf_hist = model.filter(mu_t, y, Phi, P0, return_params=["mean", "cov"])
    w_eekf_hist = eekf_hist["mean"]
    P_eekf_hist = eekf_hist["cov"]

    ### Laplace approximation
    key = random.PRNGKey(314)
    alpha = 2.0
    init_noise = 1.0
    w0 = random.multivariate_normal(key, jnp.zeros(nfeatures),
                                    jnp.eye(nfeatures) * init_noise)

    E = lambda w: -E_base(w, Phi, y, alpha) / len(y)
    res = minimize(E, w0, method="BFGS")
    w_laplace = res.x
    SN = jax.hessian(E)(w_laplace)

    ### MCMC Approximation
    chains = mcmc_logistic_posterior_sample(key, Phi, y, alpha=alpha)
    Z_mcmc = nn.sigmoid(jnp.einsum("mij,sm->sij", Phispace, chains))
    Z_mcmc = Z_mcmc.mean(axis=0)

    ### *** Ploting surface predictive distribution ***
    colors = ["black" if el else "white" for el in y]
    dict_figures = {}
    key = random.PRNGKey(31415)
    nsamples = 5000

    # FFVB surface predictive distribution
    ffvb_samples = random.multivariate_normal(key, w_ffvb, cov_ffvb, (nsamples,))
    Z_ffvb = nn.sigmoid(jnp.einsum("mij,sm->sij", Phispace, ffvb_samples))
    Z_ffvb = Z_ffvb.mean(axis=0)

    fig_ffvb, ax = plt.subplots()
    title = "EEKF  Predictive Distribution"
    plot_posterior_predictive(ax, X, Xspace, Z_ffvb, title, colors)
    dict_figures["logistic_regression_surface_ffvb"] = fig_ffvb
    plt.savefig('ssd.png')

    # EEKF surface predictive distribution
    eekf_samples = random.multivariate_normal(key, w_eekf, P_eekf, (nsamples,))
    Z_eekf = nn.sigmoid(jnp.einsum("mij,sm->sij", Phispace, eekf_samples))
    Z_eekf = Z_eekf.mean(axis=0)

    fig_eekf, ax = plt.subplots()
    title = "EEKF  Predictive Distribution"
    plot_posterior_predictive(ax, X, Xspace, Z_eekf, title, colors)
    dict_figures["logistic_regression_surface_eekf"] = fig_eekf

    # Laplace surface predictive distribution
    laplace_samples = random.multivariate_normal(key, w_laplace, SN, (nsamples,))
    Z_laplace = nn.sigmoid(jnp.einsum("mij,sm->sij", Phispace, laplace_samples))
    Z_laplace = Z_laplace.mean(axis=0)

    fig_laplace, ax = plt.subplots()
    title = "Laplace Predictive distribution"
    plot_posterior_predictive(ax, X, Xspace, Z_laplace, title, colors)
    dict_figures["logistic_regression_surface_laplace"] = fig_laplace

    # MCMC surface predictive distribution
    fig_mcmc, ax = plt.subplots()
    title = "MCMC Predictive distribution"
    plot_posterior_predictive(ax, X, Xspace, Z_mcmc, title, colors)
    dict_figures["logistic_regression_surface_mcmc"] = fig_mcmc

    ### Plot EEKF and Laplace training history
    P_eekf_hist_diag = jnp.diagonal(P_eekf_hist, axis1=1, axis2=2)
    lcolors = ["black", "tab:blue", "tab:red"]
    elements = w_eekf_hist.T, P_eekf_hist_diag.T, w_laplace, lcolors
    timesteps = jnp.arange(n_datapoints) + 1

    for k, (wk, Pk, wk_laplace, c) in enumerate(zip(*elements)):
        fig_weight_k, ax = plt.subplots()
        ax.errorbar(timesteps, wk, jnp.sqrt(Pk), c=c, label=f"$w_{k}$ online (EEKF)")
        ax.axhline(y=wk_laplace, c=c, linestyle="dotted", label=f"$w_{k}$ batch (Laplace)", linewidth=3)

        ax.set_xlim(1, n_datapoints)
        ax.legend(framealpha=0.7, loc="upper right")
        ax.set_xlabel("number samples")
        ax.set_ylabel("weights")
        plt.tight_layout()
        dict_figures[f"logistic_regression_hist_ekf_w{k}"] = fig_weight_k

    # *** Plotting posterior marginals of weights ***
    for i in range(nfeatures):
        fig_weights_marginals, ax = plt.subplots()
        mean_eekf, std_eekf = w_eekf[i], jnp.sqrt(P_eekf[i, i])
        mean_ffvb, std_ffvb = w_ffvb[i], jnp.sqrt(cov_ffvb[i, i])
        mean_laplace, std_laplace = w_laplace[i], jnp.sqrt(SN[i, i])
        mean_mcmc, std_mcmc = chains[:, i].mean(), chains[:, i].std()

        x = jnp.linspace(mean_eekf - 4 * std_eekf, mean_eekf + 4 * std_eekf, 500)
        ax.plot(x, norm.pdf(x, mean_eekf, std_eekf), label="posterior (EEKF)")
        ax.plot(x, norm.pdf(x, mean_laplace, std_laplace), label="posterior (Laplace)", linestyle="dotted")
        ax.plot(x, norm.pdf(x, mean_mcmc, std_mcmc), label="posterior (MCMC)", linestyle="dashed")
        ax.plot(x, norm.pdf(x, mean_ffvb, std_ffvb), label="posterior (FFVB)", linestyle="dashdot")

        ax.legend()
        ax.set_title(f"Posterior marginals of weights ({i})")
        # dict_figures[f"weights_marginals_w{i}"] = fig_weights_marginals
        dict_figures[f"logistic_regression_weights_marginals_w{i}"] = fig_weights_marginals

    print("MCMC weights")
    w_mcmc = chains.mean(axis=0)
    print(w_mcmc, end="\n" * 2)

    print("EEKF weights")
    print(w_eekf, end="\n" * 2)

    print("Laplace weights")
    print(w_laplace, end="\n" * 2)

    print("FFVB weights")
    print(w_ffvb, end="\n" * 2)

    dict_figures["data"] = {
        "X": X,
        "y": y,
        "Xspace": Xspace,
        "Phi": Phi,
        "Phispace": Phispace,
        "w_laplace": w_laplace,
    }

    return dict_figures


if __name__ == "__main__":
    figs = main()
    plt.savefig('smd.png')
    plt.show()
