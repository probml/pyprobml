# Compare NUTS, SGLD and Adam on sampling from a multivariate Gaussian

from typing import Dict, List, Tuple, Union
from collections import namedtuple
from typing import NamedTuple, Union, Any, Callable, Optional


from jax.random import split, normal
from jax.lax import scan
from jax import vmap, jit, random
import jax.numpy as jnp
import optax

from blackjax import nuts, stan_warmup
from sgmcmcjax.samplers import build_sgld_sampler
from sgmcmc_utils import build_nuts_sampler, build_optax_optimizer

# We use the 'quickstart' example from
# https://github.com/jeremiecoullon/SGMCMCJax

def loglikelihood(theta, x):
    return -0.5*jnp.dot(x-theta, x-theta)

def logprior(theta):
    return -0.5*jnp.dot(theta, theta)*0.01

# generate dataset
N, D = 1000, 100
key = random.PRNGKey(0)
mu_true = random.normal(key, (D,))
X_data = random.normal(key, shape=(N, D)) + mu_true


# Adam
batch_size = int(0.1*N)
opt = optax.adam(learning_rate=1e-2)
optimizer = build_optax_optimizer(opt, loglikelihood, logprior, (X_data,), batch_size)
Nsamples = 10_000
params, log_post_list = optimizer(key, Nsamples, jnp.zeros(D))
print(log_post_list.shape)
print(params.shape)
assert jnp.allclose(params, mu_true, atol=1e-1)
print('adam test passed')

# SGLD
batch_size = int(0.1*N)
dt = 1e-5
sampler = build_sgld_sampler(dt, loglikelihood, logprior, (X_data,), batch_size)
Nsamples = 10_000
samples = sampler(key, Nsamples, jnp.zeros(D))
print(samples.shape)
mu_est = jnp.mean(samples, axis=0)
assert jnp.allclose(mu_est, mu_true, atol=1e-1)
print('sgld test passed')


# NUTS / blackjax
num_warmup = 500
sampler = build_nuts_sampler(num_warmup, loglikelihood, logprior, (X_data,))
Nsamples = 10_000
samples = sampler(key, Nsamples, jnp.zeros(D))
print(samples.shape)
mu_est = jnp.mean(samples, axis=0)
assert jnp.allclose(mu_est, mu_true, atol=1e-1)
print('nuts test passed')


