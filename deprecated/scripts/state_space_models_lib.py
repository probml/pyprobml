# Library of state space models using Deepmind's distrax framework
# See: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LinearGaussianStateSpaceModel
#      https://github.com/deepmind/distrax/blob/master/distrax/_src/distributions/distribution.py
# for implementation details and specifications

# Author: Gerardo Durán-Martín (@gerdm)

# !pip install -q distrax

import superimport

import chex
import jax
import distrax
import operator
import functools
import jax.numpy as jnp
import collections.abc
import numpy as np
from jax import random
from tensorflow_probability.substrates import jax as tfp
from typing import Optional, Tuple, Union, Sequence
from distrax._src.utils import jittable

tfd = tfp.distributions
Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey
IntLike = Union[int, np.int16, np.int32, np.int64]


def MultivariateNormal(loc, covariance_matrix):
    return distrax.as_distribution(tfd.MultivariateNormalFullCovariance(loc, covariance_matrix))


def convert_seed_and_sample_shape(
    seed: Union[IntLike, PRNGKey],
    sample_shape: Union[IntLike, Sequence[IntLike]]) -> Tuple[PRNGKey, Tuple[int, ...]]:
    """Shared functionality to ensure that seeds and shapes are the right type."""
    if not isinstance(sample_shape, collections.abc.Sequence):
        sample_shape = (sample_shape,)
        sample_shape = tuple(map(int, sample_shape))

    if isinstance(seed, IntLike.__args__):
        rng = jax.random.PRNGKey(seed)
    else:  # key is of type PRNGKey
        rng = seed

    return rng, sample_shape


class LinearGaussianStateSpaceModel(jittable.Jittable):
    def __init__(self, transition_matrix, transition_noise, observation_matrix,
                 observation_noise, initial_state_prior):
        self._transition_matrix = transition_matrix
        self._transition_noise = transition_noise
        self._observation_matrix = observation_matrix
        self._observation_noise = observation_noise
        self._initial_state_prior = initial_state_prior
        self.observation_size, self.state_size = observation_matrix.shape

    @property
    def transition_matrix(self) -> Array:
        return self._transition_matrix

    @property
    def transition_noise(self) -> Array:
        return self._transition_noise

    @property
    def observation_matrix(self) -> Array:
        return self._observation_matrix

    @property
    def observation_noise(self) -> Array:
        return self._observation_noise

    @property
    def initial_state_prior(self) -> Array:
        return self._initial_state_prior

    def __kalman_step(self,
                      state: Tuple[Array, Array],
                      xt: Array
                      ) -> Tuple[Tuple[Array, Array],
                                 Tuple[Array, Array, Array, Array]]:
        mun, Sigman = state
        A = self.transition_matrix
        C = self.observation_matrix
        Q = self.transition_noise.covariance()
        R = self.observation_noise.covariance()

        I = jnp.eye(self.state_size)
        # Sigman|{n-1}
        Sigman_cond = A @ Sigman @ A.T + Q
        St = C @ Sigman_cond @ C.T + R
        Kn = Sigman_cond @ C.T @ jnp.linalg.inv(St)

        # mun|{n-1} and xn|{n-1}
        mu_update = A @ mun
        x_update = C @ mu_update

        mun = mu_update + Kn @ (xt - x_update)
        Sigman = (I - Kn @ C) @ Sigman_cond

        Q = self.transition_noise.covariance()
        R = self.observation_noise.covariance()
        # xt conditional
        mun_cond = self.transition_matrix @ mun
        xn_cond = self.observation_matrix @ mun_cond
        # St conditional
        Sigma_cond = self.transition_matrix @ Sigman @ self.transition_matrix.T + Q
        Sn = self.observation_matrix @ Sigma_cond @ self.observation_matrix.T + R

        log_likelihood = MultivariateNormal(xn_cond, Sn).log_prob(xt)


        return (mun, Sigman), (log_likelihood, mun, Sigman, mu_update, Sigman_cond)

    def __smoother_step(self,
                        state: Tuple[Array, Array],
                        elements: Tuple[Array, Array, Array, Array]):
        mut_giv_T, Sigmat_giv_T = state
        A = self.transition_matrix
        mutt, Sigmatt, mut_cond_next, Sigmat_cond_next = elements
        Jt  = Sigmatt @ A.T @ jnp.linalg.inv(Sigmat_cond_next)
        mut_giv_T = mutt + Jt @ (mut_giv_T - mut_cond_next)
        Sigmat_giv_T = Sigmatt + Jt @ (Sigmat_giv_T - Sigmat_cond_next) @ Jt.T
        return (mut_giv_T, Sigmat_giv_T), (mut_giv_T, Sigmat_giv_T)
    
    def __forward_filter(self, x: Array) -> Tuple[Array, Array, Array, Array]:
        mu0 = self.initial_state_prior.mean()
        Sigma0 = self.initial_state_prior.covariance()
        _, (log_likelihoods, filtered_means, filtered_covs, mu_cond_hist, Sigma_cond_hist) = jax.lax.scan(self.__kalman_step, (mu0, Sigma0), x)
        
        return log_likelihoods, filtered_means, filtered_covs, mu_cond_hist, Sigma_cond_hist
    
    def __backward_smoothing_pass(self, filtered_means: Array, filtered_covs: Array,
                                  mu_cond_hist: Array, Sigma_cond_hist: Array) -> Tuple[Array, Array]:
        mut_giv_T = filtered_means[-1, :]
        Sigmat_giv_T = filtered_covs[-1, :]

        elements = (filtered_means[-2::-1], filtered_covs[-2::-1, ...], mu_cond_hist[:0:-1, ...], Sigma_cond_hist[:0:-1, ...])
        _, (smoothed_means, smoothed_covs) = jax.lax.scan(self.__smoother_step, (mut_giv_T, Sigmat_giv_T), elements)
        smoothed_means = jnp.concatenate([smoothed_means[::-1, ...], mut_giv_T[None, ...]], axis=0)
        smoothed_covs = jnp.concatenate([smoothed_covs[::-1, ...], Sigmat_giv_T[None, ...]], axis=0)

        return smoothed_means, smoothed_covs

    def __sample_step(self, state_t: Array, key: PRNGKey):
        key_state, key_obs = random.split(key)
        state_mean_next = self.transition_matrix @ state_t

        state_noise = self.transition_noise.sample(seed=key_state)
        obs_noise = self.observation_noise.sample(seed=key_obs) 

        state_next = state_mean_next + state_noise
        obs_next = self.observation_matrix @ state_next + obs_noise

        return state_next, (state_next, obs_next)
    
    def _sample_n(self, key: PRNGKey, n: int, num_timesteps: int) -> Tuple[Array, Array]:
        key_init, key_next, key_step = random.split(key, 3)
        key_steps = random.split(key_step, n * num_timesteps).reshape(n, num_timesteps, -1)
        
        state_init = self.initial_state_prior.sample(seed=key_init, sample_shape=(n,))
        
        vmap_sample = jax.vmap(lambda states, keys: jax.lax.scan(self.__sample_step, states, keys))
        _, (state_samples, obs_samples) = vmap_sample(state_init, key_steps)

        return state_samples, obs_samples

    def sample(self, *,
               seed: Union[IntLike, PRNGKey],
               num_timesteps: int,
               sample_shape: Union[IntLike, Sequence[IntLike]] = ()) -> Array:
        """Samples an event.
        Parameters
        ----------
        seed: PRNG key or integer seed.
        num_timesteps: int
        sample_shape: Additional leading dimensions for sample.

        Returns
        -------
        * Array(*sample_shape, batch_shape, event_shape)
            A sample of shape `sample_shape` + `batch_shape` + `event_shape`.
        """

        rng, sample_shape = convert_seed_and_sample_shape(seed, sample_shape)
        num_samples = functools.reduce(operator.mul, sample_shape, 1)  # product

        state_samples, obs_samples = self._sample_n(rng, num_samples, num_timesteps)

        state_samples = state_samples.reshape(sample_shape + state_samples.shape[1:])
        obs_samples = obs_samples.reshape(sample_shape + obs_samples.shape[1:])

        return state_samples, obs_samples

    
    def forward_filter(self, x: Array) -> Tuple[Array, Array, Array, Array]:
        """
        Run a Kalman filter over a provided sequence of outputs.
        
        Parameters
        ----------
        x_hist: array(*batch_size, timesteps, observation_size)
            
        Returns
        -------
        * array(*batch_size, timesteps, state_size):
            Filtered means mut
        * array(*batch_size, timesteps, state_size, state_size)
            Filtered covariances Sigmat
        * array(*batch_size, timesteps, state_size)
            Filtered conditional means mut|t-1
        * array(*batch_size, timesteps, state_size, state_size)
            Filtered conditional covariances Sigmat|t-1
        """
        forward_map = jax.vmap(self.__forward_filter, 0)

        *batch_shape, timesteps, _ = x.shape
        state_mean_dims = (*batch_shape, timesteps, self.state_size)
        state_cov_dims = (*batch_shape, timesteps, self.state_size, self.state_size)

        x = x.reshape(-1, timesteps, self.observation_size)
        log_likelihoods, filtered_means, filtered_covs, mu_cond_hist, Sigma_cond_hist = forward_map(x)

        log_likelihoods = log_likelihoods.reshape(*batch_shape, timesteps)
        filtered_means = filtered_means.reshape(state_mean_dims)
        filtered_covs = filtered_covs.reshape(state_cov_dims)
        mu_cond_hist = mu_cond_hist.reshape(state_mean_dims)
        Sigma_cond_hist = Sigma_cond_hist.reshape(state_cov_dims)

        return log_likelihoods, filtered_means, filtered_covs, mu_cond_hist, Sigma_cond_hist
    
    def backward_smoothing_pass(self,
                                filtered_means: Array,
                                filtered_covs: Array,
                                mu_cond_hist: Array,
                                Sigma_cond_hist: Array) -> Tuple[Array, Array]:
        """
        Run the backward pass in Kalman smoother.
        The inputs are returned by forward_filter function

        Parameters
        ----------
        * array(*batch_size, timesteps, state_size):
            Filtered means mut
        * array(*batch_size, timesteps, state_size, state_size)
            Filtered covariances Sigmat
        * array(*batch_size, timesteps, state_size)
            Filtered conditional means mut|t-1
        * array(*batch_size, timesteps, state_size, state_size)
            Filtered conditional covariances Sigmat|t-1

        Returns
        -------
        * array(*batch_size, timesteps, state_size)
            Means of the smoothed marginal distributions
        * array(*batch_size, timesteps, state_size, state_size)
            Covariances of the smoothed marginal distributions
        """
        *batch_shape, timesteps, _ = filtered_means.shape
        state_mean_dims = (*batch_shape, timesteps, self.state_size)
        state_cov_dims = (*batch_shape, timesteps, self.state_size, self.state_size)

        filtered_means = filtered_means.reshape(-1, timesteps, self.state_size)
        filtered_covs = filtered_covs.reshape(-1, timesteps, self.state_size, self.state_size)
        mu_cond_hist = mu_cond_hist.reshape(-1, timesteps, self.state_size)
        Sigma_cond_hist = Sigma_cond_hist.reshape(-1, timesteps, self.state_size, self.state_size)

        backward_map = jax.vmap(self.__backward_smoothing_pass, 0)
        smoothed_means, smoothed_covs = backward_map(filtered_means, filtered_covs, mu_cond_hist, Sigma_cond_hist)

        smoothed_means = smoothed_means.reshape(state_mean_dims)
        smoothed_covs = smoothed_covs.reshape(state_cov_dims)

        return smoothed_means, smoothed_covs
    
    def posterior_marginals(self, x: Array) -> Array:
        """
        Run a Kalman smoother to return posterior mean and cov.
        This function only performs smoothing. If the user wants the intermediate values,
        which are returned by filtering pass `forward_filter`, one could get it by

        Parameters
        ----------
        x_hist: array(*batch_size, timesteps, observation_size)

        Returns
        -------
        * array(*batch_size, timesteps, state_size)
            Means of the smoothed marginal distributions
        * array(*batch_size, timesteps, state_size, state_size)
            Covariances of the smoothed marginal distributions
        """
        _, filtered_means, filtered_covs, mu_cond_hist, Sigma_cond_hist = self.forward_filter(x)
        smoothed_means, smoothed_covs = self.backward_smoothing_pass(filtered_means, filtered_covs,
                                                                     mu_cond_hist, Sigma_cond_hist)
        return smoothed_means, smoothed_covs

    def log_prob(self, x: Array) -> Array:
        """
        Log probability density/mass function

        Parameters
        ----------
        x_hist: array(*batch_size, timesteps, observation_size)

        Returns
        -------
        * Array(*batch_size)
            Marginal log-probabilities
        """
        log_probabilities, *_ = self.forward_filter(x)
        return log_probabilities.sum(axis=-1)
