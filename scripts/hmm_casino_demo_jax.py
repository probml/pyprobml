'''
This demo compares the Jax and Numpy version of HMMDiscrete class in terms of the speed.
Also, checks whether or not they give the same result.
Author : Aleyna Kara (@karalleyna)
'''

import time

import jax
import jax.numpy as jnp
import numpy as np

import hmm_lib as hmm
import hmm_lib_jax as hmm_jax

seed = 0
rng_key = jax.random.PRNGKey(seed)
rng_key, key_A, key_B = jax.random.split(rng_key, 3)

# state transition matrix
n_hidden, n_obs = 100, 10
A = jax.random.uniform(key_A, (n_hidden, n_hidden))
A = A / jnp.sum(A, axis=1)

# observation matrix
B = jax.random.uniform(key_B, (n_hidden, n_obs))
B = B / jnp.sum(B, axis=1).reshape((-1, 1))

n_samples = 1000
init_state_dist = jnp.ones(n_hidden) / n_hidden

seed = 0
rng_key = jax.random.PRNGKey(seed)

casino = hmm.HMMDiscrete(A, B, init_state_dist)
casino_jax = hmm_jax.HMMDiscrete(A, B, init_state_dist)

z_hist, x_hist = casino_jax.sample(n_samples, rng_key)

start = time.time()
alphas_np, loglikelihood_np  =  casino.forwards(np.array(x_hist))
print(f'Time taken by numpy version of forwards : {time.time()-start}s')

start = time.time()
alphas_jax, loglikelihood_jax  =  casino_jax.forwards(x_hist)
print(f'Time taken by jax version of forwards : {time.time()-start}s')

assert np.allclose(alphas_np, alphas_jax)
assert np.allclose(loglikelihood_np, loglikelihood_jax)

gammas_np =  casino.forwards_backwards(np.array(x_hist), alphas_np)
gammas_jax =  casino_jax.forwards_backwards(x_hist, alphas_jax)

assert np.allclose(gammas_np, gammas_jax)