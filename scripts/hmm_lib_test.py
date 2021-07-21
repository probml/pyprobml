'''
This demo compares the Jax version of Hidden Markov Model for discrete observations and general hidden markov model
in terms of the speed. It also checks whether or not the inference algorithms give the same result.
Author : Aleyna Kara (@karalleyna)
'''

import time

import jax.numpy as jnp
from jax.random import PRNGKey, split, uniform
import numpy as np

from hmm_discrete_lib import HMMJax
from hmm_discrete_lib import hmm_sample_jax, hmm_forwards_backwards_jax

from hmm_lib import HMM, hmm_forwards_backwards
import distrax

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

params = HMM(distrax.Categorical(probs=A),
                     distrax.Categorical(probs=B),
                     distrax.Categorical(probs=init_state_dist))
params_jax = HMMJax(A, B, init_state_dist)

z_hist, x_hist = hmm_sample_jax(params_jax, n_samples, rng_key)

start = time.time()
alphas, _, gammas, loglikelihood = hmm_forwards_backwards(params, x_hist, len(x_hist))
print(f'Time taken by HMM general forwards : {time.time()-start}s')
print(f'Loglikelihood of HMM general : {loglikelihood}')

start = time.time()
alphas_jax, _, gammas_jax, loglikelihood_jax = hmm_forwards_backwards_jax(params_jax, jnp.array(x_hist), len(x_hist))
print(f'Time taken by HMM discrete forwards backwards: {time.time()-start}s')
print(f'Loglikelihood of HMM discrete backwards: {loglikelihood_jax}')

assert np.allclose(alphas, alphas_jax, 8)
assert np.allclose(loglikelihood, loglikelihood_jax)
assert np.allclose(gammas, gammas_jax, 8)
