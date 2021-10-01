# make an sgmcmc-compatible interface to blakckjax
from typing import Dict, List, Tuple, Union
from collections import namedtuple
from typing import NamedTuple, Union, Any, Callable, Optional

from blackjax import nuts, stan_warmup
from jax.random import split, normal
from jax.lax import scan
from jax import vmap, jit, random
import jax.numpy as jnp
from sgmcmcjax.samplers import build_sgld_sampler

def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(carry, i):
        state, key = carry
        kernel_key, key = split(key)
        state, _ = kernel(kernel_key, state)
        return (state, key), state

    _, states = scan(one_step, (initial_state, rng_key), jnp.arange(num_samples))
    return states


def build_log_post(loglikelihood, logprior, data):
    if len(data)==1:
        batch_loglik = jit(vmap(loglikelihood, in_axes=(None, 0)))
    elif len(data)==2:
        batch_loglik = jit(vmap(loglikelihood, in_axes=(None, 0,0)))
    else:
        raise ValueError("Data must be a tuple of size 1 or 2")

    def log_post(params):
        return logprior(params) + jnp.sum(batch_loglik(params, *data), axis=0)

    return jit(log_post)

def build_nuts_sampler(num_warmup, loglikelihood, logprior, data, batchsize=None):
  # wrapper for blackjax, so it acts like other sgmcmc samplers
    log_post = build_log_post(loglikelihood, logprior, data)
    def potential(params):
         v = log_post(params)
         return -v

    def nuts_sampler(rng_key, num_samples, initial_params):
        initial_state = nuts.new_state(initial_params, potential)
 
        kernel_generator = lambda step_size, inverse_mass_matrix: jit(nuts.kernel(
            potential, step_size, inverse_mass_matrix))
        stan_key, key = split(rng_key)

        final_state, (step_size, inverse_mass_matrix), _ = stan_warmup.run(
            stan_key,
            kernel_generator,
            initial_state,
            num_warmup)

        nuts_kernel = kernel_generator(step_size, inverse_mass_matrix)

        inference_key, key = split(key)
        states = inference_loop(inference_key, nuts_kernel, final_state, num_samples)
        return states.position

    return nuts_sampler





# define model in JAX
def loglikelihood(theta, x):
    return -0.5*jnp.dot(x-theta, x-theta)

def logprior(theta):
    return -0.5*jnp.dot(theta, theta)*0.01

# generate dataset
N, D = 10_000, 100
key = random.PRNGKey(0)
mu_true = random.normal(key, (D,))
X_data = random.normal(key, shape=(N, D)) + mu_true

# build sampler
batch_size = int(0.1*N)
dt = 1e-5
sampler = build_sgld_sampler(dt, loglikelihood, logprior, (X_data,), batch_size)

# run sampler
Nsamples = 10_000
samples = sampler(key, Nsamples, jnp.zeros(D))

# test
print(samples.shape)
mu_est = jnp.mean(samples, axis=0)
print(mu_est.shape)
assert jnp.allclose(mu_est, mu_true, atol=1e-1)
print('sgld test passed')


# blackjax
num_warmup = 500
sampler = build_nuts_sampler(num_warmup, loglikelihood, logprior, (X_data,))

# run sampler
Nsamples = 10_000
samples = sampler(key, Nsamples, jnp.zeros(D))

# test
print(samples.shape)
mu_est = jnp.mean(samples, axis=0)
print(mu_est.shape)
assert jnp.allclose(mu_est, mu_true, atol=1e-1)
print('nuts test passed')