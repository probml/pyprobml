'''
Generalize hmm_discrete_lib so it can handle any kind of observation distribution (eg Gaussian, Poisson, GMM, product of
Bernoullis)
Author : Aleyna Kara(@karalleyna)
'''

from jax.random import split, PRNGKey
import jax.numpy as jnp
from jax import jit, lax, vmap

from functools import partial

'''
!pip install jax==0.2.11
!pip install jaxlib==0.1.69
!pip install ensorflow==2.5.0
!pip install tensorflow-probability==0.13.0
'''
import distrax
import flax

'''
Hidden Markov Model class used in jax implementations of inference algorithms.

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

@partial(jit, static_argnums=(1,))
def hmm_sample(params, seq_len, rng_key):
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
def hmm_forwards(params, obs_seq, length=None):
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

    trans_dist, obs_dist, init_dist = params.trans_dist, params.obs_dist, params.init_dist
    n_states = obs_dist.batch_shape[0]

    def scan_fn(carry, t):
        (alpha_prev, log_ll_prev) = carry
        alpha_n = jnp.where(t < length,
                            obs_dist.prob(obs_seq[t]) * (alpha_prev[:, None] * trans_dist.probs).sum(axis=0),
                            jnp.zeros_like(alpha_prev))

        alpha_n, cn = normalize(alpha_n)
        carry = (alpha_n, jnp.log(cn) + log_ll_prev)

        return carry, alpha_n

    # initial belief state
    alpha_0, c0 = normalize(init_dist.probs * obs_dist.prob(obs_seq[0]))

    # setup scan loop
    init_state = (alpha_0, jnp.log(c0))
    ts = jnp.arange(1, seq_len)
    carry, alpha_hist = lax.scan(scan_fn, init_state, ts)

    # post-process
    alpha_hist = jnp.vstack([alpha_0.reshape(1, n_states), alpha_hist])
    (alpha_final, log_ll) = carry
    return log_ll, alpha_hist

@jit
def hmm_backwards(params, obs_seq, length=None):
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

    trans_dist, obs_dist, init_dist = params.trans_dist, params.obs_dist, params.init_dist
    n_states = trans_dist.batch_shape[0]

    beta_t = jnp.ones((n_states,))

    def scan_fn(beta_prev, t):
        beta_t = jnp.where(t > length,
                           jnp.zeros_like(beta_prev),
                           normalize((beta_prev * obs_dist.prob(obs_seq[-t + 1]) * trans_dist.probs).sum(axis=1))[0])
        return beta_t, beta_t

    ts = jnp.arange(2, seq_len + 1)
    _, beta_hist = lax.scan(scan_fn, beta_t, ts)

    beta_hist = jnp.flip(jnp.vstack([beta_t.reshape(1, n_states), beta_hist]), axis=0)

    return beta_hist

@jit
def hmm_forwards_backwards(params, obs_seq, length=None):
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

    def gamma_t(t):
        gamma_t = jnp.where(t < length ,
                            alpha[t]* beta[t-length],
                            jnp.zeros((n_states,)))
        return gamma_t

    ll, alpha = hmm_forwards(params, obs_seq, length)
    n_states = alpha.shape[1]

    beta = hmm_backwards(params, obs_seq, length)

    ts = jnp.arange(seq_len)
    gamma = vmap(gamma_t, (0))(ts)
    #gamma = alpha * jnp.roll(beta, -seq_len + length, axis=0) #: Alternative
    gamma = vmap(lambda x: normalize(x)[0])(gamma)
    return alpha, beta, gamma, ll

@jit
def hmm_viterbi(params, obs_seq, length=None):
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
    trans_dist, obs_dist, init_dist = params.trans_dist, params.obs_dist, params.init_dist

    n_states = obs_dist.batch_shape[0]

    w0 = trans_dist.logits + init_dist.logits + obs_dist.log_prob(obs_seq[0])
    w0 = w0.max(axis=1)

    def forwards_backwards(w_prev, t):
        wn = jnp.where(t < length,
                       trans_dist.logits + obs_dist.log_prob(obs_seq[t]) + w_prev,
                       -jnp.inf + jnp.zeros_like(w_prev))
        wn = wn.max(axis=1)
        return wn, wn

    ts = jnp.arange(1, seq_len)
    _, logp_hist = lax.scan(forwards_backwards, w0, ts)
    logp_hist = jnp.vstack([w0.reshape(1, n_states), logp_hist])

    return logp_hist.argmax(axis=1)