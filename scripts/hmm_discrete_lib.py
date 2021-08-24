# Implementation of the Hidden Markov Model for discrete observations
# Author: Gerardo Duran-Martin (@gerdm), Aleyna Kara (@karalleyna)

import superimport

from dataclasses import dataclass

from numpy.random import seed
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from jax.scipy.special import logit

from functools import partial

#!pip install flax
import flax
#!pip install graphviz
from graphviz import Digraph

'''
Hidden Markov Model class used in numpy implementations of inference algorithms.
'''
@dataclass
class HMMNumpy:
   trans_mat: np.array # A : (n_states, n_states)
   obs_mat: np.array # B : (n_states, n_obs)
   init_dist: np.array # pi : (n_states)

'''
Hidden Markov Model class used in jax implementations of inference algorithms.

The functions of optimizers expect that the type of its parameters 
is pytree. So, they cannot work on a vanilla dataclass. To see more:
                https://github.com/google/jax/issues/2371

Since the flax.dataclass is registered pytree beforehand, it facilitates to use
jit, vmap and optimizers on the hidden markov model.
'''
@flax.struct.dataclass
class HMMJax:
   trans_mat: jnp.array # A : (n_states, n_states)
   obs_mat: jnp.array # B : (n_states, n_obs)
   init_dist: jnp.array # pi : (n_states)

def normalize_numpy(u, axis=0, eps=1e-15):
    '''
    Normalizes the values within the axis in a way that they sum up to 1.

    Parameters
    ----------
    u : array
    axis : int
    eps : float
        Threshold for the alpha values

    Returns
    -------
    * array
        Normalized version of the given matrix

    * array(seq_len, n_hidden) :
        The values of the normalizer
    '''
    u = np.where(u == 0, 0, np.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = np.where(c == 0, 1, c)
    return u / c, c

def normalize(u, axis=0, eps=1e-15):
    '''
    Normalizes the values within the axis in a way that they sum up to 1.

    Parameters
    ----------
    u : array
    axis : int
    eps : float
        Threshold for the alpha values

    Returns
    -------
    * array
        Normalized version of the given matrix

    * array(seq_len, n_hidden) :
        The values of the normalizer
    '''
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = jnp.where(c == 0, 1, c)
    return u / c, c

def hmm_sample_numpy(params, seq_len, random_state=0):
    '''
    Samples an observation of given length according to the defined
    hidden markov model and gives the sequence of the hidden states
    as well as the observation.

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    seq_len: array(seq_len)
        The length of the observation sequence

    random_state : int
        Seed value

    Returns
    -------
    * array(seq_len,)
        Hidden state sequence

    * array(seq_len,) :
        Observation sequence
    '''
    def sample_one_step_(hist, a, p):
        x_t = np.random.choice(a=a, p=p)
        return np.append(hist, [x_t]), x_t

    seed(random_state)

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    state_seq = np.array([], dtype=int)
    obs_seq = np.array([], dtype=int)

    latent_states = np.arange(n_states)
    obs_states = np.arange(n_obs)

    state_seq, zt = sample_one_step_(state_seq, latent_states, init_dist)
    obs_seq, xt = sample_one_step_(obs_seq, obs_states, obs_mat[zt])

    for _ in range(1, seq_len):
        state_seq, zt = sample_one_step_(state_seq, latent_states, trans_mat[zt])
        obs_seq, xt = sample_one_step_(obs_seq, obs_states, obs_mat[zt])

    return state_seq, obs_seq

@partial(jit, static_argnums=(1,))
def hmm_sample_jax(params, seq_len, rng_key):
    '''
    Samples an observation of given length according to the defined
    hidden markov model and gives the sequence of the hidden states
    as well as the observation.

    Parameters
    ----------
    params : HMMJax
        Hidden Markov Model

    seq_len: array(seq_len)
        The length of the observation sequence

    rng_key : array
        Random key of shape (2,) and dtype uint32

    Returns
    -------
    * array(seq_len,)
        Hidden state sequence

    * array(seq_len,) :
        Observation sequence
    '''
    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    initial_state = jax.random.categorical(rng_key, logits=logit(init_dist), shape=(1,))
    obs_states = jnp.arange(n_obs)

    def draw_state(prev_state, key):
        logits = logit(trans_mat[:, prev_state])
        state = jax.random.categorical(key, logits=logits.flatten(), shape=(1,))
        return state, state

    rng_key, rng_state, rng_obs = jax.random.split(rng_key, 3)
    keys = jax.random.split(rng_state, seq_len - 1)

    final_state, states = jax.lax.scan(draw_state, initial_state, keys)
    state_seq = jnp.append(jnp.array([initial_state]),states)

    def draw_obs(z, key):
        obs = jax.random.choice(key, a=obs_states, p=obs_mat[z])
        return obs

    keys = jax.random.split(rng_obs, seq_len)
    obs_seq = jax.vmap(draw_obs, in_axes=(0, 0))(state_seq, keys)

    return state_seq, obs_seq

def hmm_forwards_numpy(params, obs_seq, length):
    '''
    Calculates a belief state

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    obs_seq: array(seq_len)
        History of observable events

    Returns
    -------
    * float
        The loglikelihood giving log(p(x|model))

    * array(seq_len, n_hidden) :
        All alpha values found for each sample
    '''
    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape
    seq_len = len(obs_seq)

    alpha_hist = np.zeros((seq_len, n_states))
    ll_hist = np.zeros(seq_len)  # loglikelihood history


    alpha_n = init_dist * obs_mat[:, obs_seq[0]]
    alpha_n, cn = normalize_numpy(alpha_n)

    alpha_hist[0] = alpha_n
    ll_hist[0] = np.log(cn)

    for t in range(1, length):
        alpha_n = obs_mat[:, obs_seq[t]] * (alpha_n[:, None] * trans_mat).sum(axis=0)
        alpha_n, cn = normalize_numpy(alpha_n)

        alpha_hist[t] = alpha_n
        ll_hist[t] = np.log(cn) + ll_hist[t-1] # calculates the loglikelihood up to time t

    return ll_hist[length - 1], alpha_hist

@jit
def hmm_forwards_jax(params, obs_seq, length=None):
    '''
    Calculates a belief state

    Parameters
    ----------
    params : HMMJax
        Hidden Markov Model

    obs_seq: array(seq_len)
        History of observable events

    Returns
    -------
    * float
        The loglikelihood giving log(p(x|model))

    * array(seq_len, n_hidden) :
        All alpha values found for each sample
    '''
    seq_len = len(obs_seq)

    if length is None:
        length = seq_len

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    def scan_fn(carry, t):
        (alpha_prev, log_ll_prev) = carry
        alpha_n = jnp.where(t < length,
                            obs_mat[:, obs_seq[t]] * (alpha_prev[:, None] * trans_mat).sum(axis=0),
                            jnp.zeros_like(alpha_prev))

        alpha_n, cn = normalize(alpha_n)
        carry = (alpha_n, jnp.log(cn) + log_ll_prev)

        return carry, alpha_n

    # initial belief state
    alpha_0, c0 = normalize(init_dist * obs_mat[:, obs_seq[0]])

    # setup scan loop
    init_state = (alpha_0, jnp.log(c0))
    ts = jnp.arange(1, seq_len)
    carry, alpha_hist = lax.scan(scan_fn, init_state, ts)

    # post-process
    alpha_hist = jnp.vstack([alpha_0.reshape(1, n_states), alpha_hist])
    (alpha_final, log_ll) = carry
    return log_ll, alpha_hist

def hmm_loglikelihood_numpy(params, observations, lens):
    '''
    Finds the loglikelihood of each observation sequence sequentially.

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    observations: array(N, seq_len)
        Batch of observation sequences

    lens : array(N, seq_len)
        Consists of the valid length of each observation sequence

    Returns
    -------
    * array(N, seq_len)
        Consists of the loglikelihood of each observation sequence
    '''
    return np.array([hmm_forwards_numpy(params, obs, length)[0] for obs, length in zip(observations, lens)])

@jit
def hmm_loglikelihood_jax(params, observations, lens):
    '''
    Finds the loglikelihood of each observation sequence parallel using vmap.

    Parameters
    ----------
    params : HMMJax
        Hidden Markov Model

    observations: array(N, seq_len)
        Batch of observation sequences

    lens : array(N, seq_len)
        Consists of the valid length of each observation sequence

    Returns
    -------
    * array(N, seq_len)
        Consists of the loglikelihood of each observation sequence
    '''
    def forward_(params, x, length):
        return hmm_forwards_jax(params, x, length)[0]

    return vmap(forward_, in_axes=(None, 0, 0))(params, observations, lens)

def hmm_backwards_numpy(params, obs_seq, length=None):
    '''
    Computes the backwards probabilities

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    obs_seq: array(seq_len,)
        History of observable events

    length : array(seq_len,)
        The valid length of the observation sequence

    Returns
    -------
    * array(seq_len, n_states)
       Beta values
    '''
    seq_len = len(obs_seq)

    if length is None:
        length = seq_len

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist

    n_states, n_obs = obs_mat.shape
    beta_next = np.ones(n_states)

    beta_hist = np.zeros((seq_len, n_states))
    beta_hist[-1] = beta_next

    for t in range(2, length + 1):
        beta_next, _ = normalize_numpy((beta_next * obs_mat[:, obs_seq[-t + 1]] * trans_mat).sum(axis=1))
        beta_hist[-t] = beta_next

    return beta_hist

@jit
def hmm_backwards_jax(params, obs_seq, length=None):
    '''
    Computes the backwards probabilities

    Parameters
    ----------
    params : HMMJax
        Hidden Markov Model

    obs_seq: array(seq_len,)
        History of observable events

    length : array(seq_len,)
        The valid length of the observation sequence

    Returns
    -------
    * array(seq_len, n_states)
       Beta values
    '''
    seq_len = len(obs_seq)

    if length is None:
        length = seq_len

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    beta_t = jnp.ones((n_states,))

    def scan_fn(beta_prev, t):
        beta_t = jnp.where(t > length,
                           jnp.zeros_like(beta_prev),
                           normalize((beta_prev * obs_mat[:, obs_seq[-t + 1]] * trans_mat).sum(axis=1))[0])
        return beta_t, beta_t

    ts = jnp.arange(2, seq_len + 1)
    _, beta_hist = lax.scan(scan_fn, beta_t, ts)

    beta_hist = jnp.flip(jnp.vstack([beta_t.reshape(1, n_states), beta_hist]), axis=0)

    return beta_hist


def hmm_forwards_backwards_numpy(params, obs_seq, length=None):
    '''
    Computes, for each time step, the marginal conditional probability that the Hidden Markov Model was
    in each possible state given the observations that were made at each time step, i.e.
    P(z[i] | x[0], ..., x[num_steps - 1]) for all i from 0 to num_steps - 1

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    obs_seq: array(seq_len)
        History of observed states

    Returns
    -------
    * array(seq_len, n_states)
        Alpha values

    * array(seq_len, n_states)
        Beta values

    * array(seq_len, n_states)
        Marginal conditional probability

    * float
        The loglikelihood giving log(p(x|model))
    '''
    seq_len = len(obs_seq)
    if length is None:
        length = seq_len

    ll, alpha = hmm_forwards_numpy(params, obs_seq, length)
    beta = hmm_backwards_numpy(params, obs_seq, length)

    gamma = alpha * np.roll(beta, -seq_len + length, axis=0)
    normalizer = gamma.sum(axis=1, keepdims=True)
    gamma = gamma / np.where(normalizer==0, 1, normalizer)

    return alpha, beta, gamma, ll

@jit
def hmm_forwards_backwards_jax(params, obs_seq, length=None):
    '''
    Computes, for each time step, the marginal conditional probability that the Hidden Markov Model was
    in each possible state given the observations that were made at each time step, i.e.
    P(z[i] | x[0], ..., x[num_steps - 1]) for all i from 0 to num_steps - 1

    Parameters
    ----------
    params : HMMJax
        Hidden Markov Model

    obs_seq: array(seq_len)
        History of observed states

    Returns
    -------
    * array(seq_len, n_states)
        Alpha values

    * array(seq_len, n_states)
        Beta values

    * array(seq_len, n_states)
        Marginal conditional probability

    * float
        The loglikelihood giving log(p(x|model))
    '''
    seq_len = len(obs_seq)
    if length is None:
        length = seq_len
    def gamma_t(t):
        gamma_t = jnp.where(t < length ,
                            alpha[t]* beta[t-length],
                            jnp.zeros((n_states,)))
        return gamma_t

    ll, alpha = hmm_forwards_jax(params, obs_seq, length)
    n_states = alpha.shape[1]

    beta = hmm_backwards_jax(params, obs_seq, length)

    ts = jnp.arange(seq_len)
    gamma = vmap(gamma_t, (0))(ts)
    #gamma = alpha * jnp.roll(beta, -seq_len + length, axis=0) #: Alternative
    gamma = vmap(lambda x: normalize(x)[0])(gamma)
    return alpha, beta, gamma, ll

def hmm_viterbi_numpy(params, obs_seq):
    """
    Compute the most probable sequence of states

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    obs_seq: array(seq_len)
        History of observed states

    Returns
    -------
    * array(seq_len)
        Sequence of most MAP probable sequence of states
    """
    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape
    seq_len = len(obs_seq)

    logp_hist = np.zeros((seq_len, n_states))
    wn = np.log(trans_mat) + np.log(init_dist) + np.log(obs_mat[:, obs_seq[0]])
    wn = wn.max(axis=1)
    logp_hist[0] = wn

    for t in range(1, seq_len):
        wn = np.log(trans_mat) + np.log(obs_mat[:, obs_seq[t]]) + wn
        wn = wn.max(axis=1)
        logp_hist[t] = wn

    return logp_hist.argmax(axis=1)

@jit
def hmm_viterbi_jax(params, obs_seq, length=None):
    """
    Compute the most probable sequence of states

    Parameters
    ----------
    params : HMMJax
        Hidden Markov Model

    obs_seq: array(seq_len)
        History of observed states

    length : int
        Valid length of the observation sequence

    Returns
    -------
    * array(seq_len)
        Sequence of most MAP probable sequence of states
    """
    seq_len = len(obs_seq)

    if length is None:
        length = seq_len
    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist

    n_states, _ = obs_mat.shape

    w0 = jnp.log(trans_mat) + jnp.log(init_dist) + jnp.log(obs_mat[:, obs_seq[0]])
    w0 = w0.max(axis=1)

    def forwards_backwards(w_prev, t):
        wn = jnp.where(t < length,
                       jnp.log(trans_mat) + jnp.log(obs_mat[:, obs_seq[t]]) + w_prev,
                       -jnp.inf + jnp.zeros_like(w_prev))
        wn = wn.max(axis=1)

        return wn, wn

    ts = jnp.arange(1, seq_len)
    _, logp_hist = jax.lax.scan(forwards_backwards, w0, ts)
    logp_hist = jnp.vstack([w0.reshape(1, n_states), logp_hist])

    return logp_hist.argmax(axis=1)

def hmm_plot_graphviz(params, file_name, states=[], observations=[]):
    """
    Visualizes HMM transition matrix and observation matrix using graphhiz.

    Parameters
    ----------
    params : HMMJax or HMMNumpy
        Hidden Markov Model

    file_name : str
        Name of file which stores the output.
        The function creates file_name.pdf and file_name; the latter is a .dot text file.

    states: List(num_hidden)
        Names of hidden states

    observations: List(num_obs)
        Names of observable events

    Returns
    -------
    dot object, that can be displayed in colab
    """

    try:
        trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    except:
        raise TypeError('params must be of either HMMNumpy or HMMJax')

    n_states, n_obs = obs_mat.shape

    dot = Digraph(comment='HMM')
    if not states:
        states = [f'State {i+1}' for i in range(n_states)]
    if not observations:
        observations = [f'Obs {i+1}' for i in range(n_obs)]

    # Creates hidden state nodes
    for i, name in enumerate(states):
        table = [f'<TR><TD>{observations[j]}</TD><TD>{"%.2f" % prob}</TD></TR>' for j, prob in
                 enumerate(obs_mat[i])]
        label = f'''<<TABLE><TR><TD BGCOLOR="lightblue" COLSPAN="2">{name}</TD></TR>{''.join(table)}</TABLE>>'''
        dot.node(f's{i}', label=label)

    # Writes transition probabilities
    for i in range(n_states):
        for j in range(n_states):
            dot.edge(f's{i}', f's{j}', label=str('%.2f' % trans_mat[i, j]))
    dot.attr(rankdir='LR')
    dot.render(file_name, view=True)
    return dot
