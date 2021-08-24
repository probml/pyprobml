# Common functions that can be used for any hidden markov model type.
# Author: Aleyna Kara(@karalleyna)

import superimport

from jax import vmap, jit
from jax.random import split, randint, PRNGKey, permutation
from functools import partial
import jax.numpy as jnp

import matplotlib.pyplot as plt

@partial(jit, static_argnums=(2,))
def hmm_sample_minibatches(observations, valid_lens, batch_size, rng_key):
    '''
    Creates minibatches consists of the random permutations of the
    given observation sequences

    Parameters
    ----------
    observations : array(N, seq_len)
        All observation sequences

    valid_lens : array(N, seq_len)
        Consists of the valid length of each observation sequence

    batch_size : int
        The number of observation sequences that will be included in
        each minibatch

    rng_key : array
        Random key of shape (2,) and dtype uint32

    Returns
    -------
    * array(num_batches, batch_size, max_len)
        Minibatches
    '''
    num_train = len(observations)
    perm = permutation(rng_key, num_train)

    def create_mini_batch(batch_idx):
        return observations[batch_idx], valid_lens[batch_idx]

    num_batches = num_train // batch_size
    batch_indices = perm.reshape((num_batches, -1))
    minibatches = vmap(create_mini_batch)(batch_indices)
    return minibatches

@partial(jit, static_argnums=(1, 2, 3))
def hmm_sample_n(params, sample_fn,  n, max_len, rng_key):
    '''
    Generates n observation sequences from the given Hidden Markov Model

    Parameters
    ----------
    params : HMMNumpy or HMMJax
        Hidden Markov Model

    sample_fn :
        The sample function of the given hidden markov model

    n : int
        The total number of observation sequences

    max_len : int
        The upper bound of the length of each observation sequence. Note that the valid length of the observation
        sequence is less than or equal to the upper bound.

    rng_key : array
        Random key of shape (2,) and dtype uint32

    Returns
    -------
    * array(n, max_len)
        Observation sequences
    '''
    def sample_(params, n_samples, key):
        return sample_fn(params, n_samples, key)[1]

    rng_key, rng_lens = split(rng_key)
    lens = randint(rng_lens, (n,), minval=1, maxval=max_len + 1)
    keys = split(rng_key, n)
    observations = vmap(sample_, in_axes=(None, None, 0))(params, max_len, keys)
    return observations, lens

@jit
def pad_sequences(observations, valid_lens, pad_val=0):
    '''
    Generates n observation sequences from the given Hidden Markov Model

    Parameters
    ----------
    params : HMMNumpy or HMMJax
        Hidden Markov Model

    observations : array(N, seq_len)
        All observation sequences

    valid_lens : array(N, seq_len)
        Consists of the valid length of each observation sequence

    pad_val : int
        Value that the invalid observable events of the observation sequence will be replaced

    Returns
    -------
    * array(n, max_len)
        Ragged dataset
    '''
    def pad(seq, len):
        idx = jnp.arange(1, seq.shape[0] + 1)
        return jnp.where(idx <= len, seq, pad_val)
    ragged_dataset = vmap(pad, in_axes=(0, 0))(observations, valid_lens), valid_lens
    return ragged_dataset

def plot_loss_curve(loss_values, title=""):
    '''
    Plots loss curve

    Parameters
    ----------
    loss_values : array
    title : str
    '''
    plt.title(title)
    plt.xlabel("Number of Iterations")
    plt.plot(loss_values)
    plt.show()