'''
This demo compares the log space version of Hidden Markov Model for discrete observations and general hidden markov model
in terms of the speed. It also checks whether or not the inference algorithms give the same result.
Author : Aleyna Kara (@karalleyna)
'''

import superimport

import time

import jax.numpy as jnp
from jax.random import PRNGKey, split, uniform
import numpy as np


from hmm_lib_log import HMM, hmm_forwards_backwards_log, hmm_viterbi_log, hmm_sample_log

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

hmm = HMM(trans_dist=distrax.Categorical(probs=A),
          obs_dist=distrax.Categorical(probs=B),
          init_dist=distrax.Categorical(probs=init_state_dist))

hmm_distrax = distrax.HMM(trans_dist=distrax.Categorical(probs=A),
          obs_dist=distrax.Categorical(probs=B),
          init_dist=distrax.Categorical(probs=init_state_dist))

z_hist, x_hist = hmm_sample_log(hmm, n_samples, rng_key)

start = time.time()
alphas, _, gammas, loglikelihood = hmm_distrax.forward_backward(x_hist, len(x_hist))
print(f'Time taken by Forwards Backwards function of HMM general: {time.time()-start}s')
print(f'Loglikelihood found by HMM general: {loglikelihood}')

start = time.time()
alphas_log, _, gammas_log, loglikelihood_log = hmm_forwards_backwards_log(hmm, x_hist, len(x_hist))
print(f'Time taken by Forwards Backwards function of HMM Log Space Version: {time.time()-start}s')
print(f'Loglikelihood found by HMM General Log Space Version: {loglikelihood_log}')

assert np.allclose(jnp.log(alphas), alphas_log, 8)
assert np.allclose(loglikelihood, loglikelihood_log)
assert np.allclose(jnp.log(gammas), gammas_log, 8)

# Test for the hmm_viterbi_log. This test is based on https://github.com/deepmind/distrax/blob/master/distrax/_src/utils/hmm_test.py
loc = jnp.array([0.0, 1.0, 2.0, 3.0])
scale = jnp.array(0.25)
initial = jnp.array([0.25, 0.25, 0.25, 0.25])
trans = jnp.array([[0.9, 0.1, 0.0, 0.0],
                   [0.1, 0.8, 0.1, 0.0],
                   [0.0, 0.1, 0.8, 0.1],
                   [0.0, 0.0, 0.1, 0.9]])

observations = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 3.0, 2.9, 2.8, 2.7, 2.6])

model = HMM(
    init_dist=distrax.Categorical(probs=initial),
    trans_dist=distrax.Categorical(probs=trans),
    obs_dist=distrax.Normal(loc, scale))

inferred_states = hmm_viterbi_log(model, observations)
expected_states = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]

assert np.allclose(inferred_states, expected_states)

length = 7
inferred_states = hmm_viterbi_log(model, observations, length)
expected_states = [0, 0, 0, 0, 1, 2, 3, -1, -1, -1]
assert np.allclose(inferred_states, expected_states)

