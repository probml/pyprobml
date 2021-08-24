# Trains hidden markov models with discrete observations using Gradient Descent in a stateless way.
# Author : Aleyna Kara(@karalleyna)

import superimport

import itertools

import jax
from jax.nn import softmax
from jax.random import PRNGKey, split, normal
from jax import jit, vmap

from hmm_utils import hmm_sample_minibatches
from hmm_discrete_lib import HMMJax, hmm_loglikelihood_jax

opt_init, opt_update, get_params = None, None, None

def init_random_params(sizes, rng_key):
    '''
    Initializes the components of HMM from normal distibution

    Parameters
    ----------
    sizes: List
      Consists of number of hidden states and observable events, respectively

    rng_key : array
        Random key of shape (2,) and dtype uint32

    Returns
    -------
    * array(num_hidden, num_hidden)
      Transition probability matrix

    * array(num_hidden, num_obs)
      Emission probability matrix

    * array(1, num_hidden)
      Initial distribution probabilities
    '''
    num_hidden, num_obs = sizes
    rng_key, rng_a, rng_b, rng_pi = split(rng_key, 4)
    return HMMJax(normal(rng_a, (num_hidden, num_hidden)),
                  normal(rng_b, (num_hidden, num_obs)),
                  normal(rng_pi, (num_hidden,)))


@jit
def loss_fn(params, batch, lens):
    '''
    Objective function of hidden markov models for discrete observations. It returns the mean of the negative
    loglikelihood of the sequence of observations

    Parameters
    ----------
    params : HMMJax
        Hidden Markov Model

    batch: array(N, max_len)
        Minibatch consisting of observation sequences

    lens : array(N, seq_len)
        Consists of the valid length of each observation sequence in the minibatch

    Returns
    -------
    * float
        The mean negative loglikelihood of the minibatch
    '''
    params_soft = HMMJax(softmax(params.trans_mat, axis=1),
                         softmax(params.obs_mat, axis=1),
                         softmax(params.init_dist))
    return -hmm_loglikelihood_jax(params_soft, batch, lens).mean()


@jit
def update(i, opt_state, batch, lens):
    '''
    Objective function of hidden markov models for discrete observations. It returns the mean of the negative
    loglikelihood of the sequence of observations

    Parameters
    ----------
    i : int
        Specifies the current iteration

    opt_state : OptimizerState

    batch: array(N, max_len)
        Minibatch consisting of observation sequences

    lens : array(N, seq_len)
        Consists of the valid length of each observation sequence in the minibatch

    Returns
    -------
    * OptimizerState

    * float
        The mean negative loglikelihood of the minibatch, i.e. loss value for the current iteration.
    '''
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(loss_fn)(params, batch, lens)
    return opt_update(i, grads, opt_state), loss


def fit(observations, lens, num_hidden, num_obs, batch_size, optimizer,rng_key=None, num_epochs=1):
    '''
    Trains the HMM model with the given number of hidden states and observations via any optimizer.

    Parameters
    ----------
    observations: array(N, seq_len)
        All observation sequences

    lens : array(N, seq_len)
        Consists of the valid length of each observation sequence

    num_hidden : int
        The number of hidden state

    num_obs : int
        The number of observable events

    batch_size : int
        The number of observation sequences that will be included in each minibatch

    optimizer : jax.experimental.optimizers.Optimizer
        Optimizer that is used during training

    num_epochs : int
        The total number of iterations

    Returns
    -------
    * HMMJax
        Hidden Markov Model

    * array
      Consists of training losses
    '''
    global opt_init, opt_update, get_params

    if rng_key is None:
        rng_key = PRNGKey(0)

    rng_init, rng_iter = split(rng_key)
    params = init_random_params([num_hidden, num_obs], rng_init)
    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(params)
    itercount = itertools.count()

    def epoch_step(opt_state, key):

        def train_step(opt_state, params):
            batch, length = params
            opt_state, loss = update(next(itercount), opt_state, batch, length)
            return opt_state, loss

        batches, valid_lens = hmm_sample_minibatches(observations, lens, batch_size, key)
        params = (batches, valid_lens)
        opt_state, losses = jax.lax.scan(train_step, opt_state, params)
        return opt_state, losses.mean()

    epochs = split(rng_iter, num_epochs)
    opt_state, losses = jax.lax.scan(epoch_step, opt_state, epochs)

    losses = losses.flatten()

    params = get_params(opt_state)
    params = HMMJax(softmax(params.trans_mat, axis=1),
                    softmax(params.obs_mat, axis=1),
                    softmax(params.init_dist))
    return params, losses