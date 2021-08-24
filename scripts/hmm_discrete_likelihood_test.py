'''
This demo shows how to create variable size dataset and then it creates mini-batches from this dataset so that it
calculates the likelihood of each observation sequence in every batch using vmap.
Author : Aleyna Kara (@karalleyna)
'''

import superimport

from jax import vmap, jit
from jax.random import split, randint, PRNGKey
import jax.numpy as jnp

import hmm_utils
from hmm_discrete_lib import hmm_sample_jax, hmm_loglikelihood_numpy, hmm_loglikelihood_jax
from hmm_discrete_lib import HMMNumpy, HMMJax
import numpy as np

def loglikelihood_numpy(params_numpy, batches, lens):
  return np.vstack([hmm_loglikelihood_numpy(params_numpy, batch, l) for batch, l in zip(batches, lens)])

def loglikelihood_jax(params_jax, batches, lens):
  return vmap(hmm_loglikelihood_jax, in_axes=(None, 0, 0))(params_jax, batches, lens)

# state transition matrix
A = jnp.array([
    [0.95, 0.05],
    [0.10, 0.90]
])

# observation matrix
B = jnp.array([
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], # fair die
    [1/10, 1/10, 1/10, 1/10, 1/10, 5/10] # loaded die
])

pi = jnp.array([1, 1]) / 2

params_numpy= HMMNumpy(np.array(A), np.array(B), np.array(pi))
params_jax = HMMJax(A, B, pi)

seed = 0
rng_key = PRNGKey(seed)
rng_key, rng_sample = split(rng_key)

n_obs_seq, batch_size, max_len = 15, 5, 10

observations, lens = hmm_utils.hmm_sample_n(params_jax,
                                            hmm_sample_jax,
                                            n_obs_seq, max_len,
                                            rng_sample)

observations, lens = hmm_utils.pad_sequences(observations, lens)

rng_key, rng_batch = split(rng_key)
batches, lens = hmm_utils.hmm_sample_minibatches(observations,
                                                 lens,
                                                 batch_size,
                                                 rng_batch)

ll_numpy = loglikelihood_numpy(params_numpy, np.array(batches), np.array(lens))
ll_jax = loglikelihood_jax(params_jax, batches, lens)

assert np.allclose(ll_numpy, ll_jax)
print(f'Loglikelihood {ll_numpy}')