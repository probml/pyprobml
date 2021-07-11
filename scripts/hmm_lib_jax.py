# Implementation of the Hidden Markov Model for discrete observations with Jax.
# This file is based on https://github.com/probml/pyprobml/blob/master/scripts/hmm_lib.py
# Author: Gerardo Duran-Martin (@gerdm), Aleyna Kara (@karalleyna)

from jax import lax
import jax
import jax.numpy as jnp

class HMMDiscrete:
    def __init__(self, A, px, pi):
        """
        This class simulates a Hidden Markov Model with
        categorical distribution

        Parameters
        ----------
        A: array(state_size, state_size)
            State transition matrix
        px: array(state_size, observation_size)
            Matrix of conditional categorical probabilities
            of obsering the ith category
        pi: array(state_size)
            Array of initial-state probabilities
        """
        self.A = A
        self.px = px
        self.pi = pi
        self.state_size, self.observation_size = px.shape

    def sample(self, n_samples, rng_key):
        rng_key, key_x, key_z = jax.random.split(rng_key, 3)
        latent_states = jnp.arange(self.state_size)
        obs_states = jnp.arange(self.observation_size)

        zt = jax.random.choice(key_z, latent_states, p=self.pi)
        xt = jax.random.choice(key_x, obs_states, p=self.px[zt])

        z_hist = jnp.array([zt])
        x_hist = jnp.array([xt])

        for t in range(1, n_samples):
            rng_key, key_x, key_z = jax.random.split(rng_key, 3)
            zt = jax.random.choice(key_z, latent_states, p=self.A[zt])
            xt = jax.random.choice(key_x, obs_states, p=self.px[zt])
            z_hist = jnp.append(z_hist, jnp.array([zt]))
            x_hist = jnp.append(x_hist, jnp.array([xt]))

        return z_hist, x_hist

    def forwards(self, x_hist):
        """
        Calculates a belief state
        Parameters
        ----------
        x_hist: array(n_samples)
            History of observed states

        Returns
        -------
        * array(n_samples, n_hidden) :
            All alpha values found for each sample
        * float
            The loglikelihood giving log(p(x|model))
        """
        n_samples = len(x_hist)

        alpha_hist = jnp.zeros((n_samples, self.state_size))
        c_elements = jnp.zeros(n_samples)

        alpha_n = self.pi * self.px[:, x_hist[0]]
        cn = alpha_n.sum()
        alpha_n = alpha_n / cn

        alpha_hist = jax.ops.index_update(alpha_hist, jax.ops.index[0, :], alpha_n)
        c_elements = jax.ops.index_update(c_elements, jax.ops.index[0], cn) # normalization constants

        def scan_fn(alpha_with_norm_const, t):
            alpha_hist, c_elements = alpha_with_norm_const
            alpha_n = self.px[:, x_hist[t]] * (alpha_hist[t - 1, :].reshape((-1, 1)) * self.A).sum(axis=0)
            cn = alpha_n.sum()
            alpha_n = alpha_n / cn

            alpha_hist = jax.ops.index_update(alpha_hist, jax.ops.index[t, : ], alpha_n)
            c_elements = jax.ops.index_update(c_elements, jax.ops.index[t], cn)

            return (alpha_hist, c_elements), jnp.zeros((0,))

        (alpha_hist, c_elements), _ = lax.scan(scan_fn, (alpha_hist, c_elements), jnp.arange(1, n_samples))
        return alpha_hist, jnp.sum(jnp.log(c_elements))

    def backwards_filtering(self, x_hist):
        n_samples = len(x_hist)
        beta_next = jnp.ones(self.state_size)

        beta_hist = jnp.zeros((n_samples, self.state_size))
        beta_hist = jax.ops.index_update(beta_hist, jax.ops.index[-1, :], beta_next)

        def scan_fn(beta_hist, t):
            beta_next = (beta_hist[-t + 1] * self.px[:, x_hist[-t + 1]] * self.A).sum(axis=1)
            beta_hist = jax.ops.index_update(beta_hist, jax.ops.index[-t, :], beta_next / beta_next.sum())
            return beta_hist, jnp.zeros((0,))

        beta_hist, _ = lax.scan(scan_fn, beta_hist, jnp.arange(2, n_samples + 1))
        return beta_hist

    def forwards_backwards(self, x_hist, alpha_hist=None, beta_hist=None):
        if alpha_hist is None:
            alpha_hist, _ = self.forwards(x_hist)
        if beta_hist is None:
            beta_hist = self.backwards_filtering(x_hist)
        gamma = alpha_hist * beta_hist
        return gamma / gamma.sum(axis=1).reshape((-1, 1))

    def map_state(self, x_hist):
        """
        Compute the most probable sequence of states
        Parameters
        ----------
        x_hist: array(n_samples)
            History of observed states

        Returns
        -------
        * array(n_samples)
            Sequence of most MAP probable sequence of states
        """
        n_samples = len(x_hist)
        wn = jnp.log(self.A) + jnp.log(self.pi) + jnp.log(self.px[:, x_hist[0]])
        wn = wn.max(axis=1)
        logp_hist = jnp.array(wn)

        for t in range(1, n_samples):
            wn = jnp.log(self.A) + jnp.log(self.px[:, x_hist[t]]) + wn
            wn = wn.max(axis=1)
            logp_hist = jnp.vstack((logp_hist, jnp.array(wn)))
        return logp_hist.argmax(axis=1)# Implementation of the Hidden Markov Model for discrete observations with Jax.
# This file is based on https://github.com/probml/pyprobml/blob/master/scripts/hmm_lib.py
# Author: Gerardo Duran-Martin (@gerdm), Aleyna Kara (@karalleyna)

from jax import lax
import jax
import jax.numpy as jnp

class HMMDiscrete:
    def __init__(self, A, px, pi):
        """
        This class simulates a Hidden Markov Model with
        categorical distribution

        Parameters
        ----------
        A: array(state_size, state_size)
            State transition matrix
        px: array(state_size, observation_size)
            Matrix of conditional categorical probabilities
            of obsering the ith category
        pi: array(state_size)
            Array of initial-state probabilities
        """
        self.A = A
        self.px = px
        self.pi = pi
        self.state_size, self.observation_size = px.shape

    def sample(self, n_samples, rng_key):
        rng_key, key_x, key_z = jax.random.split(rng_key, 3)
        latent_states = jnp.arange(self.state_size)
        obs_states = jnp.arange(self.observation_size)

        zt = jax.random.choice(key_z, latent_states, p=self.pi)
        xt = jax.random.choice(key_x, obs_states, p=self.px[zt])

        z_hist = jnp.array([zt])
        x_hist = jnp.array([xt])

        for t in range(1, n_samples):
            rng_key, key_x, key_z = jax.random.split(rng_key, 3)
            zt = jax.random.choice(key_z, latent_states, p=self.A[zt])
            xt = jax.random.choice(key_x, obs_states, p=self.px[zt])
            z_hist = jnp.append(z_hist, jnp.array([zt]))
            x_hist = jnp.append(x_hist, jnp.array([xt]))

        return z_hist, x_hist

    def forwards(self, x_hist):
        """
        Calculates a belief state
        Parameters
        ----------
        x_hist: array(n_samples)
            History of observed states

        Returns
        -------
        * array(n_samples, n_hidden) :
            All alpha values found for each sample
        * float
            The loglikelihood giving log(p(x|model))
        """
        n_samples = len(x_hist)

        alpha_hist = jnp.zeros((n_samples, self.state_size))
        c_elements = jnp.zeros(n_samples)

        alpha_n = self.pi * self.px[:, x_hist[0]]
        cn = alpha_n.sum()
        alpha_n = alpha_n / cn

        alpha_hist = jax.ops.index_update(alpha_hist, jax.ops.index[0, :], alpha_n)
        c_elements = jax.ops.index_update(c_elements, jax.ops.index[0], cn) # normalization constants

        def scan_fn(alpha_with_norm_const, t):
            alpha_hist, c_elements = alpha_with_norm_const
            alpha_n = self.px[:, x_hist[t]] * (alpha_hist[t - 1, :].reshape((-1, 1)) * self.A).sum(axis=0)
            cn = alpha_n.sum()
            alpha_n = alpha_n / cn

            alpha_hist = jax.ops.index_update(alpha_hist, jax.ops.index[t, : ], alpha_n)
            c_elements = jax.ops.index_update(c_elements, jax.ops.index[t], cn)

            return (alpha_hist, c_elements), jnp.zeros((0,))

        (alpha_hist, c_elements), _ = lax.scan(scan_fn, (alpha_hist, c_elements), jnp.arange(1, n_samples))
        return alpha_hist, jnp.sum(jnp.log(c_elements))

    def backwards_filtering(self, x_hist):
        n_samples = len(x_hist)
        beta_next = jnp.ones(self.state_size)

        beta_hist = jnp.zeros((n_samples, self.state_size))
        beta_hist = jax.ops.index_update(beta_hist, jax.ops.index[-1, :], beta_next)

        def scan_fn(beta_hist, t):
            beta_next = (beta_hist[-t + 1] * self.px[:, x_hist[-t + 1]] * self.A).sum(axis=1)
            beta_hist = jax.ops.index_update(beta_hist, jax.ops.index[-t, :], beta_next / beta_next.sum())
            return beta_hist, jnp.zeros((0,))

        beta_hist, _ = lax.scan(scan_fn, beta_hist, jnp.arange(2, n_samples + 1))
        return beta_hist

    def forwards_backwards(self, x_hist, alpha_hist=None, beta_hist=None):
        if alpha_hist is None:
            alpha_hist, _ = self.forwards(x_hist)
        if beta_hist is None:
            beta_hist = self.backwards_filtering(x_hist)
        gamma = alpha_hist * beta_hist
        return gamma / gamma.sum(axis=1).reshape((-1, 1))

    def map_state(self, x_hist):
        """
        Compute the most probable sequence of states
        Parameters
        ----------
        x_hist: array(n_samples)
            History of observed states

        Returns
        -------
        * array(n_samples)
            Sequence of most MAP probable sequence of states
        """
        n_samples = len(x_hist)
        wn = jnp.log(self.A) + jnp.log(self.pi) + jnp.log(self.px[:, x_hist[0]])
        wn = wn.max(axis=1)
        logp_hist = jnp.array(wn)

        for t in range(1, n_samples):
            wn = jnp.log(self.A) + jnp.log(self.px[:, x_hist[t]]) + wn
            wn = wn.max(axis=1)
            logp_hist = jnp.vstack((logp_hist, jnp.array(wn)))
        return logp_hist.argmax(axis=1)