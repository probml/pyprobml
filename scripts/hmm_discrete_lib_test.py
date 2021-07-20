'''
This demo compares the Jax and Numpy version of forwards-backwards algorithm in terms of the speed.
Also, checks whether or not they give the same result.
Author : Aleyna Kara (@karalleyna)
'''

import time

import jax.numpy as jnp
from jax.random import PRNGKey, split, uniform
import numpy as np

from hmm_discrete_lib import HMMNumpy, HMMJax
from hmm_discrete_lib import hmm_sample_jax, hmm_forwards_backwards_numpy, hmm_forwards_backwards_jax


seed = 0
rng_key = PRNGKey(seed)
rng_key, key_A, key_B = split(rng_key, 3)

# state transition matrix
n_hidden, n_obs = 100, 10
A = uniform(key_A, (n_hidden, n_hidden))
A = A / jnp.sum(A, axis=1)

# observation matrix
B = uniform(key_B, (n_hidden, n_obs))
B = B / jnp.sum(B, axis=1).reshape((-1, 1))

n_samples = 1000
init_state_dist = jnp.ones(n_hidden) / n_hidden

seed = 0
rng_key = PRNGKey(seed)

params_numpy = HMMNumpy(A, B, init_state_dist)
params_jax = HMMJax(A, B, init_state_dist)

z_hist, x_hist = hmm_sample_jax(params_jax, n_samples, rng_key)

start = time.time()
alphas_np, _, gammas_np, loglikelihood_np = hmm_forwards_backwards_numpy(params_numpy, x_hist, len(x_hist))
print(f'Time taken by numpy version of forwards backwards : {time.time()-start}s')

start = time.time()
alphas_jax, _, gammas_jax, loglikelihood_jax = hmm_forwards_backwards_jax(params_jax, jnp.array(x_hist), len(x_hist))
print(f'Time taken by JAX version of forwards backwards: {time.time()-start}s')

assert np.allclose(alphas_np, alphas_jax)
assert np.allclose(loglikelihood_np, loglikelihood_jax)
assert np.allclose(gammas_np, gammas_jax)