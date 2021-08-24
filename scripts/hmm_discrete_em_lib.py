# The implementation of Baulm-Welch algorithm for Hidden Markov Models with discrete observations in a stateless way.
# Author : Aleyna Kara(@karalleyna)

import superimport

from scipy.special import softmax
import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.ops import index_update, index
from jax.random import PRNGKey

import hmm_discrete_lib as hmm
from dataclasses import dataclass

@dataclass
class PriorsNumpy:
   trans_pseudo_counts: np.array
   obs_pseudo_counts: np.array
   init_pseudo_counts: np.array

@dataclass
class PriorsJax:
   trans_pseudo_counts: jnp.array
   obs_pseudo_counts: jnp.array
   init_pseudo_counts: jnp.array

def init_random_params_numpy(sizes, random_state):
    '''

    Initializes the components of HMM from normal distibution

    Parameters
    ----------
    sizes: List
        Consists of the number of hidden states and observable events, respectively

    random_state : int
        Seed value

    Returns
    -------
    * HMMNumpy
        Hidden Markov Model
    '''
    num_hidden, num_obs = sizes
    np.random.seed(random_state)
    return hmm.HMMNumpy(softmax(np.random.randn(num_hidden, num_hidden), axis=1),
                    softmax(np.random.randn(num_hidden, num_obs), axis=1),
                    softmax(np.random.randn(num_hidden)))


def init_random_params_jax(sizes, rng_key):
    '''

    Initializes the components of HMM from uniform distibution

    Parameters
    ----------
    sizes: List
        Consists of number of hidden states and observable events, respectively

    rng_key : array
        Random key of shape (2,) and dtype uint32

    Returns
    -------
    * HMMJax
        Hidden Markov Model
    '''
    num_hidden, num_obs = sizes
    rng_key, rng_a, rng_b, rng_pi = jax.random.split(rng_key, 4)
    return hmm.HMMJax(jax.nn.softmax(jax.random.normal(rng_a, (num_hidden, num_hidden)), axis=1),
                  jax.nn.softmax(jax.random.normal(rng_b, (num_hidden, num_obs)), axis=1),
                  jax.nn.softmax(jax.random.normal(rng_pi, (num_hidden,))))

def compute_expected_trans_counts_numpy(params, alpha, beta, obs, T):
    '''
    Computes the expected transition counts by summing ksi_{jk} for the observation given for all states j and k.
    ksi_{jk} for any time t in [0, T-1] can be calculated as the multiplication of the probability of ending
    in state j at t, the probability of starting in state k at t+1, the transition probability a_{jk} and b_{k obs[t+1]}.
    Note that ksi[t] is normalized so that the probabilities sums up to 1 for each time t in [0, T-1].

    Parameters
    ----------
    params: HMMNumpy
       Hidden Markov Model

    alpha : array
        A matrix of shape (seq_len, n_states)

    beta : array
        A matrix of shape (seq_len, n_states)

    obs : array
        One observation sequence

    T : int
        The valid length of observation sequence

    Returns
    ----------

    * array
        The matrix of shape (n_states, n_states) representing expected transition counts given obs o.
    '''
    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    AA = np.zeros((n_states, n_states))  # AA[,j,k] = sum_t p(z(t)=j, z(t+1)=k|obs)

    for t in range(T - 1):
        ksi = alpha[t] * trans_mat.T * beta[t + 1] * obs_mat[:, obs[t + 1]]
        normalizer = ksi.sum()
        ksi /= 1 if normalizer==0 else ksi.sum()
        AA += ksi.T
    return AA

def compute_expected_trans_counts_jax(params, alpha, beta, observations):
    '''
    Computes the expected transition counts by summing ksi_{jk} for the observation given for all states j and k.
    ksi_{jk} for any time t in [0, T-1] can be calculated as the multiplication of the probability of ending
    in state j at t, the probability of starting in state k at t+1, the transition probability a_{jk} and b_{k obs[t+1]}.
    Note that ksi[t] is normalized so that the probabilities sums up to 1 for each time t in [0, T-1].

    Parameters
    ----------
    params: HMMJax
       Hidden Markov Model

    alpha : array
        A matrix of shape (num_obs_seq, seq_len, n_states) in which each row stands for the alpha of the
        corresponding observation sequence

    beta : array
        A matrix of shape (num_obs_seq, seq_len, n_states) in which each row stands for the beta of the
        corresponding observation sequence

    observations : array
        All observation sequences

    Returns
    ----------
    * array
        A matrix of shape (n_states, n_states) representing expected transition counts
    '''
    def ksi_(trans_mat, obs_mat, alpha, beta, obs):
        return (alpha * trans_mat.T * beta * obs_mat[:, obs]).T

    def count_(trans_mat, obs_mat, alpha, beta, obs):
        # AA[,j,k] = sum_t p(z(t)=j, z(t+1)=k|obs)
        AA = vmap(ksi_, in_axes=(None, None, 0, 0, 0))(trans_mat, obs_mat, alpha[:-1], beta[1:], obs[1:])
        return AA

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist

    trans_counts = vmap(count_, in_axes=(None, None, 0, 0, 0))(trans_mat, obs_mat, alpha, beta, observations)

    trans_count_normalizer = jnp.sum(trans_counts, axis=[2, 3], keepdims=True)
    trans_count_normalizer = jnp.where(trans_count_normalizer == 0, 1, trans_count_normalizer)

    trans_counts = jnp.sum(trans_counts / trans_count_normalizer, axis=1)
    trans_counts = jnp.sum(trans_counts, axis=0)

    return trans_counts


def compute_expected_obs_counts_numpy(gamma, obs, T, n_states, n_obs):
    '''
    Computes the expected observation count for each observation o by summing the probability of being at any of the
    states for each time t.
    Parameters
    ----------
    gamma : array
        A matrix of shape (seq_len, n_states)

    obs : array
        An array of shape (seq_len,)

    T : int
        The valid length of observation sequence

    n_states : int
        The number of hidden states

    n_obs : int
        The number of observable events

    Returns
    ----------
    * array
        A matrix of shape (n_states, n_obs) representing expected observation counts given observation sequence.
    '''
    BB = np.zeros((n_states, n_obs))
    for t in range(T):
        o = obs[t]
        BB[:, o] += gamma[t]
    return BB

def compute_expected_obs_counts_jax(gamma, obs, n_states, n_obs):
    '''
    Computes the expected observation count for each observation o by summing the probability of being at any of the
    states for each time t.
    Parameters
    ----------
    gamma : array
        A matrix of shape (seq_len, n_states)

    obs : array
        An array of shape (seq_len,)

    n_states : int
        The number of hidden states

    n_obs : int
        The number of observable events

    Returns
    ----------
    * array
        A matrix of shape (n_states, n_obs) representing expected observation counts given observation sequence.
    '''
    def scan_fn(BB, elems):
        o, g = elems
        BB = index_update(BB, index[:, o], BB[:, o] + g)
        return BB, jnp.zeros((0,))

    BB = jnp.zeros((n_states, n_obs))
    BB, _ = jax.lax.scan(scan_fn, BB, (obs, gamma))
    return BB

def hmm_e_step_numpy(params, observations, valid_lengths):
    '''

    Calculates the the expectation of the complete loglikelihood over the distribution of
    observations given the current parameters

    Parameters
    ----------
    params: HMMNumpy
       Hidden Markov Model

    observations : array
        All observation sequences

    valid_lengths : array
        Valid lengths of each observation sequence

    Returns
    ----------
    * array
        A matrix of shape (n_states, n_states) representing expected transition counts

    * array
        A matrix of shape (n_states, n_obs) representing expected observation counts

    * array
        An array of shape (n_states,) representing expected initial counts calculated from summing gamma[0] of each
        observation sequence

    * float
        The sum of the likelihood, p(o | lambda) where lambda stands for (trans_mat, obs_mat, init_dist) triple, for
        each observation sequence o.
    '''
    N, _ = observations.shape

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    trans_counts = np.zeros((n_states, n_states))
    obs_counts = np.zeros((n_states, n_obs))
    init_counts = np.zeros((n_states))

    loglikelihood = 0

    for obs, valid_len in zip(observations, valid_lengths):
        alpha, beta, gamma, ll = hmm.hmm_forwards_backwards_numpy(params, obs, valid_len)
        trans_counts = trans_counts + compute_expected_trans_counts_numpy(params, alpha, beta, obs, valid_len)
        obs_counts = obs_counts + compute_expected_obs_counts_numpy(gamma, obs, valid_len, n_states, n_obs)
        init_counts = init_counts + gamma[0]
        loglikelihood += ll

    return trans_counts, obs_counts, init_counts, loglikelihood

def hmm_e_step_jax(params, observations, valid_lengths):
    '''

    Calculates the the expectation of the complete loglikelihood over the distribution of
    observations given the current parameters

    Parameters
    ----------
    params: HMMJax
       Hidden Markov Model

    observations : array
        All observation sequences

    valid_lengths : array
        Valid lengths of each observation sequence

    Returns
    ----------
    * array
        A matrix of shape (n_states, n_states) representing expected transition counts

    * array
        A matrix of shape (n_states, n_obs) representing expected observation counts

    * array
        An array of shape (n_states,) representing expected initial state counts calculated from summing gamma[0]
        of each observation sequence

    * float
        The sum of the likelihood, p(o | lambda) where lambda stands for (trans_mat, obs_mat, init_dist) triple, for
        each observation sequence o.
    '''
    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    alpha, beta, gamma, ll = vmap(hmm.hmm_forwards_backwards_jax, in_axes=(None, 0, 0))(params, observations, valid_lengths)
    trans_counts = compute_expected_trans_counts_jax(params, alpha, beta, observations)

    obs_counts = vmap(compute_expected_obs_counts_jax, in_axes=(0, 0, None, None))(gamma, observations, n_states, n_obs)
    obs_counts = jnp.sum(obs_counts, axis=0)

    init_counts = jnp.sum(gamma[:, 0, :], axis=0)
    loglikelihood = jnp.sum(ll)

    return trans_counts, obs_counts, init_counts, loglikelihood

def hmm_m_step_numpy(counts, priors=None):
    '''

    Recomputes new parameters from A, B and pi using max likelihood.

    Parameters
    ----------
    counts: tuple
        Consists of expected transition counts, expected observation counts, and expected initial state counts,
        respectively.

    priors : PriorsNumpy

    Returns
    ----------
    * HMMNumpy
        Hidden Markov Model

    '''
    trans_counts, obs_counts, init_counts = counts

    if priors is not None:
        trans_counts = trans_counts + priors.trans_pseudo_counts
        obs_counts = obs_counts + priors.obs_pseudo_count
        init_counts = init_counts + priors.init_pseudo_counts

    A = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    B = obs_counts / obs_counts.sum(axis=1, keepdims=True)
    pi = init_counts / init_counts.sum()

    return hmm.HMMNumpy(A, B, pi)

def hmm_m_step_jax(counts, priors=None):
    '''

    Recomputes new parameters from A, B and pi using max likelihood.

    Parameters
    ----------
    counts: tuple
        Consists of expected transition counts, expected observation counts, and expected initial state counts,
        respectively.

    priors : PriorsNumpy

    Returns
    ----------
    * HMMJax
        Hidden Markov Model

    '''
    trans_counts, obs_counts, init_counts = counts

    if priors is not None:
        trans_counts = trans_counts + priors.trans_pseudo_counts
        obs_counts = obs_counts + priors.obs_pseudo_count
        init_counts = init_counts + priors.init_pseudo_counts

    A_denom = trans_counts.sum(axis=1, keepdims=True)
    A = trans_counts / jnp.where(A_denom == 0, 1, A_denom)

    B_denom = obs_counts.sum(axis=1, keepdims=True)
    B = obs_counts / jnp.where(B_denom == 0, 1, B_denom)

    pi = init_counts / init_counts.sum()
    return hmm.HMMJax(A, B, pi)

def hmm_em_numpy(observations, valid_lengths, n_hidden=None, n_obs=None,
                 init_params=None, priors=None, num_epochs=1, random_state=None):
    '''
    Implements Baum–Welch algorithm which is used for finding its components, A, B and pi.

    Parameters
    ----------
    observations: array
        All observation sequences

    valid_lengths : array
        Valid lengths of each observation sequence

    n_hidden : int
        The number of hidden states

    n_obs : int
        The number of observable events

    init_params : HMMNumpy
        Initial Hidden Markov Model

    priors : PriorsNumpy
        Priors for the components of Hidden Markov Model

    num_epochs : int
        Number of times model will be trained

    random_state: int
        Seed value

    Returns
    ----------
    * HMMNumpy
        Trained Hidden Markov Model

    * array
        Negative loglikelihoods each of which can be interpreted as the loss value at the current iteration.
    '''

    if random_state is None:
        random_state = 0

    if init_params is None:
        try:
            init_params = init_random_params_numpy([n_hidden, n_obs], random_state)
        except:
            raise ValueError("n_hidden and n_obs should be specified when init_params was not given.")

    neg_loglikelihoods = []
    params = init_params

    for _ in range(num_epochs):
        trans_counts, obs_counts, init_counts, ll = hmm_e_step_numpy(params, observations, valid_lengths)
        neg_loglikelihoods.append(-ll)
        params = hmm_m_step_numpy([trans_counts, obs_counts, init_counts], priors)

    return params, neg_loglikelihoods

def hmm_em_jax(observations, valid_lengths, n_hidden=None, n_obs=None,
               init_params=None, priors=None, num_epochs=1, rng_key=None):
    '''
    Implements Baum–Welch algorithm which is used for finding its components, A, B and pi.

    Parameters
    ----------
    observations: array
        All observation sequences

    valid_lengths : array
        Valid lengths of each observation sequence

    n_hidden : int
        The number of hidden states

    n_obs : int
        The number of observable events

    init_params : HMMJax
        Initial Hidden Markov Model

    priors : PriorsJax
        Priors for the components of Hidden Markov Model

    num_epochs : int
        Number of times model will be trained

    rng_key : array
        Random key of shape (2,) and dtype uint32

    Returns
    ----------
    * HMMJax
        Trained Hidden Markov Model

    * array
        Negative loglikelihoods each of which can be interpreted as the loss value at the current iteration.
    '''
    if rng_key is None:
        rng_key = PRNGKey(0)

    if init_params is None:
        try:
            init_params = init_random_params_jax([n_hidden, n_obs], rng_key=rng_key)
        except:
            raise ValueError("n_hidden and n_obs should be specified when init_params was not given.")

    epochs = jnp.arange(num_epochs)

    def train_step(params, epoch):
        trans_counts, obs_counts, init_counts, ll = hmm_e_step_jax(params, observations, valid_lengths)
        params = hmm_m_step_jax([trans_counts, obs_counts, init_counts], priors)
        return params, -ll

    final_params, neg_loglikelihoods = jax.lax.scan(train_step, init_params, epochs)

    return final_params, neg_loglikelihoods