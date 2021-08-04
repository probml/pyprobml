'''
Generalizes hmm_discrete_lib so it can handle any kind of observation distribution (eg Gaussian, Poisson, GMM, product of
Bernoullis). It is based on https://github.com/probml/pyprobml/blob/master/scripts/hmm_lib.py
and operates within the log space.
Author : Aleyna Kara(@karalleyna)
'''

from jax.random import split
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.nn import logsumexp, log_softmax, one_hot

from functools import partial
'''
!pip install jax==0.2.11
!pip install jaxlib==0.1.69
!pip install tensorflow==2.5.0
!pip install tensorflow-probability==0.13.0
'''
import flax
import distrax

'''
Hidden Markov Model class in which trans_dist and init_dist are categorical-like
distribution from distrax, and obs_dist is any instance of distrax.Distribution.

The functions of optimizers expect that the type of its parameters 
is pytree. So, they cannot work on a vanilla dataclass. To see more:
                https://github.com/google/jax/issues/2371

Since the flax.dataclass is registered pytree beforehand, it facilitates to use
jit, vmap and optimizers on the hidden markov model.
'''
@flax.struct.dataclass
class HMM:
  trans_dist: distrax.Distribution
  obs_dist: distrax.Distribution
  init_dist: distrax.Distribution

def logdotexp(u, v, axis=-1):
    '''
    Calculates jnp.log(jnp.exp(u) * jnp.exp(v)) in a stable way.
    Parameters
    ----------
    u : array
    v : array
    axis : int
    Returns
    -------
    * array
        Logarithm of the Hadamard product of u and v
    '''
    max_u = jnp.max(u, axis=axis, keepdims=True)
    max_v = jnp.max(v, axis=axis, keepdims=True)

    diff_u = jnp.nan_to_num(u - max_u, -jnp.inf)
    diff_v = jnp.nan_to_num(v - max_v, -jnp.inf)

    u_dot_v = jnp.log(jnp.exp(diff_u) * jnp.exp(diff_v))
    u_dot_v = u_dot_v + max_u + max_v

    return u_dot_v


def log_normalize(u, axis=-1):
    '''
    Normalizes the values within the axis in a way that the exponential of each values within the axis
    sums up to 1.
    Parameters
    ----------
    u : array
    axis : int
    Returns
    -------
    * array
        The Log of normalized version of the given matrix
    * array(seq_len, n_hidden) :
        The values of the normalizer
    '''
    c = logsumexp(u, axis=axis)
    return jnp.where(u == -jnp.inf, -jnp.inf, u - c), c


@partial(jit, static_argnums=(1,))
def hmm_sample_log(params, seq_len, rng_key):
    '''
    Samples an observation of given length according to the defined
    hidden markov model and gives the sequence of the hidden states
    as well as the observation.
    Parameters
    ----------
    params : HMM
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
    trans_dist, obs_dist, init_dist = params.trans_dist, params.obs_dist, params.init_dist

    rng_key, rng_init = split(rng_key)
    initial_state = init_dist.sample(seed=rng_init)

    def draw_state(prev_state, key):
        state = trans_dist.sample(seed=key)[prev_state]
        return state, state

    rng_key, rng_state, rng_obs = split(rng_key, 3)
    keys = split(rng_state, seq_len - 1)
    final_state, states = lax.scan(draw_state, initial_state, keys)
    states = jnp.append(initial_state, states)

    def draw_obs(z, key):
        return obs_dist.sample(seed=key)[z]

    keys = split(rng_obs, seq_len)
    obs_seq = vmap(draw_obs, in_axes=(0, 0))(states, keys)

    return states, obs_seq


@jit
def hmm_forwards_log(params, obs_seq, length=None):
    '''
    Calculates a belief state
    Parameters
    ----------
    params : HMM
        Hidden Markov Model
    obs_seq: array(seq_len)
        History of observable events
    Returns
    -------
    * float
        The loglikelihood giving log(p(x|model))
    * array(seq_len, n_hidden) :
        Log of alpha values
    '''
    seq_len = len(obs_seq)

    if length is None:
        length = seq_len

    trans_dist, obs_dist, init_dist = params.trans_dist, params.obs_dist, params.init_dist
    n_states = obs_dist.batch_shape[0]

    def scan_fn(carry, t):
        (alpha_prev, log_ll_prev) = carry
        alpha_n = jnp.where(t < length,
                            obs_dist.log_prob(obs_seq[t]) + logsumexp(
                                logdotexp(alpha_prev[:, None], trans_dist.logits), axis=0),
                            -jnp.inf + jnp.zeros_like(alpha_prev))

        alpha_n, cn = log_normalize(alpha_n)
        carry = (alpha_n, cn + log_ll_prev)

        return carry, alpha_n

    # initial belief state
    alpha_0, c0 = log_normalize(init_dist.logits + obs_dist.log_prob(obs_seq[0]))
    # setup scan loop
    init_state = (alpha_0, c0)
    ts = jnp.arange(1, seq_len)

    carry, alpha_hist = lax.scan(scan_fn, init_state, ts)

    # post-process
    alpha_hist = jnp.vstack([alpha_0.reshape(1, n_states), alpha_hist])
    (alpha_final, log_ll) = carry
    return log_ll, alpha_hist


@jit
def hmm_backwards_log(params, obs_seq, length=None):
    '''
    Computes the backwards probabilities
    Parameters
    ----------
    params : HMM
        Hidden Markov Model
    obs_seq: array(seq_len,)
        History of observable events
    length : array(seq_len,)
        The valid length of the observation sequence
    Returns
    -------
    * array(seq_len, n_states)
       Log of beta values
    '''
    seq_len = len(obs_seq)

    if length is None:
        length = seq_len

    trans_dist, obs_dist, init_dist = params.trans_dist, params.obs_dist, params.init_dist
    n_states = trans_dist.batch_shape[0]

    beta_t = jnp.zeros((n_states,))

    def scan_fn(beta_prev, t):
        beta_t = jnp.where(t > length,
                           -jnp.inf + jnp.zeros_like(beta_prev),
                           log_normalize(logsumexp(beta_prev + obs_dist.log_prob(obs_seq[-t + 1]) + trans_dist.logits,
                                                axis=1))[0])
        return beta_t, beta_t

    ts = jnp.arange(2, seq_len + 1)
    _, beta_hist = lax.scan(scan_fn, beta_t, ts)

    beta_hist = jnp.flip(jnp.vstack([beta_t.reshape(1, n_states), beta_hist]), axis=0)

    return beta_hist


@jit
def hmm_forwards_backwards_log(params, obs_seq, length=None):
    '''
    Computes, for each time step, the marginal conditional probability that the Hidden Markov Model was
    in each possible state given the observations that were made at each time step, i.e.
    P(z[i] | x[0], ..., x[num_steps - 1]) for all i from 0 to num_steps - 1
    Parameters
    ----------
    params : HMM
        Hidden Markov Model
    obs_seq: array(seq_len)
        History of observed states
    Returns
    -------
    * array(seq_len, n_states)
        The log of alpha values
    * array(seq_len, n_states)
        The log of beta values
    * array(seq_len, n_states)
        The log of marginal conditional probability
    * float
        The loglikelihood giving log(p(x|model))
    '''
    seq_len = len(obs_seq)

    if length is None:
        length = seq_len

    def gamma_t(t):
        gamma_t = jnp.where(t < length,
                            alpha[t] + beta[t - length],
                            jnp.zeros((n_states,)))
        return gamma_t

    ll, alpha = hmm_forwards_log(params, obs_seq, length)
    n_states = alpha.shape[1]

    beta = hmm_backwards_log(params, obs_seq, length)

    ts = jnp.arange(seq_len)
    gamma = vmap(gamma_t, (0))(ts)
    # gamma = alpha * jnp.roll(beta, -seq_len + length, axis=0) #: Alternative
    gamma = vmap(lambda x: log_normalize(x, axis=0)[0])(gamma)
    return alpha, beta, gamma, ll


@jit
def hmm_viterbi_log(params, obs_seq, length=None):
    '''
    Computes, for each time step, the marginal conditional probability that the Hidden Markov Model was
    in each possible state given the observations that were made at each time step, i.e.
    P(z[i] | x[0], ..., x[num_steps - 1]) for all i from 0 to num_steps - 1
    It is based on https://github.com/deepmind/distrax/blob/master/distrax/_src/utils/hmm.py

    Parameters
    ----------
    params : HMM
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

    trans_dist, obs_dist, init_dist = params.trans_dist, params.obs_dist, params.init_dist

    trans_log_probs = log_softmax(trans_dist.logits)
    init_log_probs = log_softmax(init_dist.logits)

    n_states = obs_dist.batch_shape[0]

    first_log_prob = init_log_probs + obs_dist.log_prob(obs_seq[0])

    if seq_len == 1:
        return jnp.expand_dims(jnp.argmax(first_log_prob), axis=0)

    def viterbi_forward(prev_logp, t):
        obs_logp = obs_dist.log_prob(obs_seq[t])

        logp = jnp.where(t <= length,
                         prev_logp[..., None] + trans_log_probs + obs_logp[..., None, :],
                         -jnp.inf + jnp.zeros_like(trans_log_probs))

        max_logp_given_successor = jnp.where(t <= length, jnp.max(logp, axis=-2), prev_logp)
        most_likely_given_successor = jnp.where(t <= length, jnp.argmax(logp, axis=-2), -1)

        return max_logp_given_successor, most_likely_given_successor

    ts = jnp.arange(1, seq_len)
    final_log_prob, most_likely_sources = lax.scan(viterbi_forward, first_log_prob, ts)

    most_likely_initial_given_successor = jnp.argmax(
        trans_log_probs + first_log_prob, axis=-2)

    most_likely_sources = jnp.concatenate([
        jnp.expand_dims(most_likely_initial_given_successor, axis=0),
        most_likely_sources], axis=0)

    def viterbi_backward(state, t):
        state = jnp.where(t <= length,
                          jnp.sum(most_likely_sources[t] * one_hot(state, n_states)).astype(jnp.int64),
                          state)
        most_likely = jnp.where(t <= length, state, -1)
        return state, most_likely

    final_state = jnp.argmax(final_log_prob)
    _, most_likely_path = lax.scan(viterbi_backward, final_state, ts, reverse=True)

    final_state = jnp.where(length == seq_len, final_state, -1)

    return jnp.append(most_likely_path, final_state)