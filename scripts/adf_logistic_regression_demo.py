# Example of the training of a logistic regression
# model using Assumed Density Filter (ADF)
# Dependencies:
#   !pip install git+https://github.com/blackjax-devs/blackjax.git
#   !pip install jax_cosmo

# Author: Gerardo Durán-Martín (@gerdm)

import jax
import jax.numpy as jnp
import blackjax.rwmh as mh
import matplotlib.pyplot as plt
from sklearn.datasets import make_biclusters
from jax import random
from jax_cosmo.scipy import integrate


def sigmoid(z): return jnp.exp(z) / (1 + jnp.exp(z))

def log_sigmoid(z): return z - jnp.log(1 + jnp.exp(z))

def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

n_datapoints = 20
m = 2
X, rows, cols = make_biclusters((n_datapoints, m), 2, noise=0.6, random_state=314, minval=-3, maxval=3)
# whether datapoints belong to class 1
y = rows[0] * 1.0

alpha = 1.0
Phi = jnp.c_[jnp.ones(n_datapoints)[:, None], X]
N, M = Phi.shape

def E(w):
    an = Phi @ w
    log_an = log_sigmoid(an)
    log_likelihood_term = y * log_an + (1 - y) * jnp.log(1 - sigmoid(an))
    prior_term = alpha * w @ w / 2

    return prior_term - log_likelihood_term.sum()

key = random.PRNGKey(314)
init_noise = 1.0
w0 = random.multivariate_normal(key, jnp.zeros(M), jnp.eye(M) * init_noise)

sigma_mcmc = 0.8
initial_state = mh.new_state(w0, E)

mcmc_kernel = mh.kernel(E, jnp.ones(M) * sigma_mcmc)
mcmc_kernel = jax.jit(mcmc_kernel)

n_samples = 5_000
burnin = 300
key_init = jax.random.PRNGKey(0)
states = inference_loop(key_init, mcmc_kernel, initial_state, n_samples)

chains = states.position[burnin:, :]
nsamp, _ = chains.shape

fig, ax = plt.subplots(1, 3, figsize=(8, 2))
for i, axi in enumerate(ax):
    axi.plot(states.position[:, i])
    axi.set_title(f"$w_{i}$")
    axi.axvline(x=burnin, c="tab:red")
    axi.set_xlabel("step")
plt.tight_layout()

pml.savefig("log-reg-mcmc-chains.pdf")
plt.show()