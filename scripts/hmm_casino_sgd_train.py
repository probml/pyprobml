'''
This demo does MAP estimation of an HMM using gradient-descent algorithm applied to the log marginal likelihood.
It includes

1. Mini Batch Gradient Descent
2. Full Batch Gradient Descent
3. Stochastic Gradient Descent

Author: Aleyna Kara(@karalleyna)
'''
import superimport

from jax.random import split, randint, PRNGKey
import jax.numpy as jnp

import matplotlib.pyplot as plt

from hmm_discrete_lib import HMMJax
from hmm_discrete_lib import hmm_sample_jax, hmm_plot_graphviz

from hmm_sgd_lib import fit
from hmm_utils import pad_sequences, hmm_sample_n
from jax.experimental import optimizers

# state transition matrix
A = jnp.array([
    [0.95, 0.05],
    [0.10, 0.90]])

# observation matrix
B = jnp.array([
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], # fair die
    [1/10, 1/10, 1/10, 1/10, 1/10, 5/10] # loaded die
])

pi = jnp.array([1, 1]) / 2

casino = HMMJax(A, B, pi)
num_hidden, num_obs = 2, 6

seed = 0
rng_key = PRNGKey(seed)
rng_key, rng_sample = split(rng_key)

n_obs_seq, max_len = 4, 5000
num_epochs = 400

observations, lens = pad_sequences(*hmm_sample_n(casino, hmm_sample_jax, n_obs_seq, max_len, rng_sample))
optimizer = optimizers.momentum(step_size=1e-3, mass=0.95)

# Mini Batch Gradient Descent
batch_size = 2
params_mbgd,losses_mbgd =  fit(observations,
                               lens,
                               num_hidden,
                               num_obs,
                               batch_size,
                               optimizer,
                               rng_key=None,
                               num_epochs=num_epochs)

# Full Batch Gradient Descent
batch_size = n_obs_seq
params_fbgd,losses_fbgd =  fit(observations,
                               lens,
                               num_hidden,
                               num_obs,
                               batch_size,
                               optimizer,
                               rng_key=None,
                               num_epochs=num_epochs)

# Stochastic Gradient Descent
batch_size = 1
params_sgd,losses_sgd =  fit(observations,
                             lens,
                             num_hidden,
                             num_obs,
                             batch_size,
                             optimizer,
                             rng_key=None,
                             num_epochs=num_epochs)

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 8))

losses = [ losses_sgd, losses_mbgd, losses_fbgd]
titles = [ 'Stochastic Gradient Descent', 'Mini Batch Gradient Descent', 'Full Batch Gradient Descent']

for ax, (loss, title) in zip(axes.flat, zip(losses, titles)):
    ax.plot(loss)
    ax.set_title(f'{title}')
plt.show()

hmm_plot_graphviz(params_sgd, "hmm_casino_sgd_train")