# Jax implementation of a Linear Dynamical System
# Author:  Gerardo Durán-Martín (@gerdm), Aleyna Kara(@karalleyna)

import superimport

import jax
from jax import vmap
from jax.random import multivariate_normal, split
from jax.lax import Precision
import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax.lax import scan
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class KalmanFilter:
    """
    Implementation of the Kalman Filtering and Smoothing
    procedure of a Linear Dynamical System with known parameters.

    This class exemplifies the use of Kalman Filtering assuming
    the model parameters are known.

    Parameters
    ----------
    A: array(state_size, state_size)
        Transition matrix
    C: array(observation_size, state_size)
        Observation matrix
    Q: array(state_size, state_size)
        Transition covariance matrix
    R: array(observation_size, observation_size)
        Observation covariance
    mu0: array(state_size)
        Mean of initial configuration
    Sigma0: array(state_size, state_size) or 0
        Covariance of initial configuration. If value is set
        to zero, the initial state will be completely determined
        by mu0
    timesteps: int
        Total number of steps to sample
    """

    def __init__(self, A, C, Q, R, mu0, Sigma0=None, timesteps=15):
        self.A = A
        self.__C = C
        self.Q = Q
        self.R = R

        self.mu0 = mu0
        self.Sigma0 = Sigma0
        self.timesteps = timesteps
        self.state_size, _ = A.shape

    def C(self, t):
        if callable(self.__C):
            return self.__C(t)
        else:
            return self.__C

    def sample(self, key, n_samples=1, sample_intial_state=False):
        """
        Simulate a run of n_sample independent stochastic
        linear dynamical systems

        Parameters
        ----------
        key: jax.random.PRNGKey
            Seed of initial random states
        n_samples: int
            Number of independent linear systems with shared dynamics (optional)
        sample_initial_state: bool
            Whether to sample from an initial state or sepecified

        Returns
        -------
        * array(n_samples, timesteps, state_size):
            Simulation of Latent states
        * array(n_samples, timesteps, observation_size):
            Simulation of observed states
        """
        key_z1, key_system_noise, key_obs_noise = split(key, 3)
        if not sample_intial_state:
            state_t = self.mu0 * jnp.ones((n_samples, self.state_size))
        else:
            state_t = multivariate_normal(key_z1, self.mu0, self.Sigma0, (n_samples,))

        # Generate all future noise terms
        zeros_state = jnp.zeros(self.state_size)
        observation_size = self.timesteps if isinstance(self.R, int) else self.R.shape[0]
        zeros_obs = jnp.zeros(observation_size)

        system_noise = multivariate_normal(key_system_noise, zeros_state, self.Q, (self.timesteps, n_samples))
        obs_noise = multivariate_normal(key_obs_noise, zeros_obs, self.R, (self.timesteps, n_samples))

        obs_t = jnp.einsum("ij,sj->si", self.C(0), state_t) + obs_noise[0]

        def __sample(state, carry):
            system_noise_t, obs_noise_t, t = carry
            state_new = jnp.einsum("ij,sj->si", self.A, state) + system_noise_t
            obs_new = jnp.einsum("ij,sj->si", self.C(t), state_new) + obs_noise_t
            return state_new, (state_new, obs_new)

        timesteps = jnp.arange(1, self.timesteps)
        _, (state_hist, obs_hist) = jax.lax.scan(__sample, state_t, (system_noise[1:], obs_noise[1:], timesteps))
        state_hist = jnp.swapaxes(jnp.vstack([state_t[None, ...], state_hist]), 0, 1)
        obs_hist = jnp.swapaxes(jnp.vstack([obs_t[None, ...], obs_hist]), 0, 1)

        if n_samples == 1:
            state_hist = state_hist[0, ...]
            obs_hist = obs_hist[0, ...]
        return state_hist, obs_hist

    def kalman_step(self, state, xt):
        mun, Sigman, t = state
        I = jnp.eye(self.state_size)

        # Sigman|{n-1}
        Sigman_cond = self.A @ Sigman @ self.A.T + self.Q
        St = self.C(t) @ Sigman_cond @ self.C(t).T + self.R
        Kn = Sigman_cond @ self.C(t).T @ inv(St)

        # mun|{n-1} and xn|{n-1}
        mu_update = self.A @ mun
        x_update = self.C(t) @ mu_update

        mun = mu_update + Kn @ (xt - x_update)
        Sigman = (I - Kn @ self.C(t)) @ Sigman_cond
        t = t + 1

        return (mun, Sigman, t), (mun, Sigman, mu_update, Sigman_cond)

    def __smoother_step(self, state, elements):
        mut_giv_T, Sigmat_giv_T = state
        mutt, Sigmatt, mut_cond_next, Sigmat_cond_next = elements
        Jt = Sigmatt @ self.A.T @ inv(Sigmat_cond_next)
        mut_giv_T = mutt + Jt @ (mut_giv_T - mut_cond_next)
        Sigmat_giv_T = Sigmatt + Jt @ (Sigmat_giv_T - Sigmat_cond_next) @ Jt.T
        return (mut_giv_T, Sigmat_giv_T), (mut_giv_T, Sigmat_giv_T)

    def __kalman_filter(self, x_hist):
        """
        Compute the online version of the Kalman-Filter, i.e,
        the one-step-ahead prediction for the hidden state or the
        time update step

        Parameters
        ----------
        x_hist: array(timesteps, observation_size)

        Returns
        -------
        * array(timesteps, state_size):
            Filtered means mut
        * array(timesteps, state_size, state_size)
            Filtered covariances Sigmat
        * array(timesteps, state_size)
            Filtered conditional means mut|t-1
        * array(timesteps, state_size, state_size)
            Filtered conditional covariances Sigmat|t-1
        """
        _, (mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist) = jax.lax.scan(self.kalman_step,
                                                                               (self.mu0, self.Sigma0, 0), x_hist)
        return mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist

    def __kalman_smoother(self, mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist):
        """
        Compute the offline version of the Kalman-Filter, i.e,
        the kalman smoother for the hidden state.
        Note that we require to independently run the kalman_filter function first

        Parameters
        ----------
        mu_hist: array(timesteps, state_size):
            Filtered means mut
        Sigma_hist: array(timesteps, state_size, state_size)
            Filtered covariances Sigmat
        mu_cond_hist: array(timesteps, state_size)
            Filtered conditional means mut|t-1
        Sigma_cond_hist: array(timesteps, state_size, state_size)
            Filtered conditional covariances Sigmat|t-1

        Returns
        -------
        * array(timesteps, state_size):
            Smoothed means mut
        * array(timesteps, state_size, state_size)
            Smoothed covariances Sigmat
        """
        timesteps, _ = mu_hist.shape
        state_size, _ = self.A.shape
        mu_hist_smooth = jnp.zeros((timesteps, state_size))
        Sigma_hist_smooth = jnp.zeros((timesteps, state_size, state_size))

        mut_giv_T = mu_hist[-1, :]
        Sigmat_giv_T = Sigma_hist[-1, :]

        elements = (
            mu_hist[-2::-1], Sigma_hist[-2::-1, ...], mu_cond_hist[1:][::-1, ...], Sigma_cond_hist[1:][::-1, ...])
        _, (mu_hist_smooth, Sigma_hist_smooth) = jax.lax.scan(self.__smoother_step, (mut_giv_T, Sigmat_giv_T), elements)
        mu_hist_smooth = jnp.concatenate([mu_hist_smooth[::-1, ...], mut_giv_T[None, ...]], axis=0)
        Sigma_hist_smooth = jnp.concatenate([Sigma_hist_smooth[::-1, ...], Sigmat_giv_T[None, ...]], axis=0)

        return mu_hist_smooth, Sigma_hist_smooth

    def filter(self, x_hist):
        """
        Compute the online version of the Kalman-Filter, i.e,
        the one-step-ahead prediction for the hidden state or the
        time update step.
        Note that x_hist can optionally be of dimensionality two,
        This corresponds to different samples of the same underlying
        Linear Dynamical System

        Parameters
        ----------
        x_hist: array(n_samples?, timesteps, observation_size)

        Returns
        -------
        * array(n_samples?, timesteps, state_size):
            Filtered means mut
        * array(n_samples?, timesteps, state_size, state_size)
            Filtered covariances Sigmat
        * array(n_samples?, timesteps, state_size)
            Filtered conditional means mut|t-1
        * array(n_samples?, timesteps, state_size, state_size)
            Filtered conditional covariances Sigmat|t-1
        """
        has_one_sim = False
        if x_hist.ndim == 2:
            x_hist = x_hist[None, ...]
            has_one_sim = True
        kalman_map = jax.vmap(self.__kalman_filter, 0)
        mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = kalman_map(x_hist)
        if has_one_sim:
            mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = mu_hist[0, ...], Sigma_hist[0, ...], mu_cond_hist[
                0, ...], Sigma_cond_hist[0, ...]
        return mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist

    def smooth(self, mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist):
        """
        Compute the offline version of the Kalman-Filter, i.e,
        the kalman smoother for the state space.
        Note that we require to independently run the kalman_filter function first.
        Note that the mean terms can optionally be of dimensionality two.
        Similarly, the covariance terms can optinally be of dimensionally three.
        This corresponds to different samples of the same underlying
        Linear Dynamical System

        Parameters
        ----------
        mu_hist: array(n_samples?, timesteps, state_size):
            Filtered means mut
        Sigma_hist: array(n_samples?, timesteps, state_size, state_size)
            Filtered covariances Sigmat
        mu_cond_hist: array(n_samples?, timesteps, state_size)
            Filtered conditional means mut|t-1
        Sigma_cond_hist: array(n_samples?, timesteps, state_size, state_size)
            Filtered conditional covariances Sigmat|t-1

        Returns
        -------
        * array(n_samples?, timesteps, state_size):
            Smoothed means mut
        * array(timesteps?, state_size, state_size)
            Smoothed covariances Sigmat
        """
        has_one_sim = False
        if mu_hist.ndim == 2:
            mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = mu_hist[None, ...], Sigma_hist[None, ...], \
                                                                 mu_cond_hist[None, ...], Sigma_cond_hist[None, ...]
            has_one_sim = True
        smoother_map = jax.vmap(self.__kalman_smoother, 0)
        mu_hist_smooth, Sigma_hist_smooth = smoother_map(mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist)
        if has_one_sim:
            mu_hist_smooth, Sigma_hist_smooth = mu_hist_smooth[0, ...], Sigma_hist_smooth[0, ...]
        return mu_hist_smooth, Sigma_hist_smooth


class KalmanFilterNoiseEstimation:
    """
    Implementation of the Kalman Filtering and Smoothing
    procedure of a Linear Dynamical System with known parameters.
    This class exemplifies the use of Kalman Filtering assuming
    the model parameters are known.
    Parameters
    ----------
    A: array(state_size, state_size)
        Transition matrix
    C: array(observation_size, state_size)
        Observation matrix
    Q: array(state_size, state_size)
        Transition covariance matrix
    R: array(observation_size, observation_size)
        Observation covariance
    mu0: array(state_size)
        Mean of initial configuration
    Sigma0: array(state_size, state_size) or 0
        Covariance of initial configuration. If value is set
        to zero, the initial state will be completely determined
        by mu0
    timesteps: int
        Total number of steps to sample
    """

    def __init__(self, A, Q, mu0, Sigma0, v0, tau0, update_fn=None):
        self.A = A
        self.Q = Q
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        self.v = v0
        self.tau = tau0
        self.__update_fn = update_fn

    def update(self, state, bel, *args):
        if self.__update_fn is None:
            return bel
        else:
            return self.__update_fn(state, bel, *args)

    def kalman_step(self, state, xt):
        mu, Sigma, v, tau = state
        x, y = xt

        mu_cond = jnp.matmul(self.A, mu, precision=Precision.HIGHEST)
        Sigmat_cond = jnp.matmul(jnp.matmul(self.A, Sigma, precision=Precision.HIGHEST), self.A,
                                 precision=Precision.HIGHEST) + self.Q

        e_k = y - x.T @ mu_cond
        s_k = x.T @ Sigmat_cond @ x + 1
        Kt = (Sigmat_cond @ x) / s_k

        mu = mu + e_k * Kt
        Sigma = Sigmat_cond - jnp.outer(Kt, Kt) * s_k

        v_update = v + 1
        tau = (v * tau + (e_k * e_k) / s_k) / v_update

        return mu, Sigma, v_update, tau

    def __kalman_filter(self, x_hist):
        """
        Compute the online version of the Kalman-Filter, i.e,
        the one-step-ahead prediction for the hidden state or the
        time update step
        Parameters
        ----------
        x_hist: array(timesteps, observation_size)
        Returns
        -------
        * array(timesteps, state_size):
            Filtered means mut
        * array(timesteps, state_size, state_size)
            Filtered covariances Sigmat
        * array(timesteps, state_size)
            Filtered conditional means mut|t-1
        * array(timesteps, state_size, state_size)
            Filtered conditional covariances Sigmat|t-1
        """
        _, (mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist) = scan(self.kalman_step,
                                                                       (self.mu0, self.Sigma0, 0), x_hist)
        return mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist

    def filter(self, x_hist):
        """
        Compute the online version of the Kalman-Filter, i.e,
        the one-step-ahead prediction for the hidden state or the
        time update step.
        Note that x_hist can optionally be of dimensionality two,
        This corresponds to different samples of the same underlying
        Linear Dynamical System
        Parameters
        ----------
        x_hist: array(n_samples?, timesteps, observation_size)
        Returns
        -------
        * array(n_samples?, timesteps, state_size):
            Filtered means mut
        * array(n_samples?, timesteps, state_size, state_size)
            Filtered covariances Sigmat
        * array(n_samples?, timesteps, state_size)
            Filtered conditional means mut|t-1
        * array(n_samples?, timesteps, state_size, state_size)
            Filtered conditional covariances Sigmat|t-1
        """
        has_one_sim = False
        if x_hist.ndim == 2:
            x_hist = x_hist[None, ...]
            has_one_sim = True
        kalman_map = vmap(self.__kalman_filter, 0)
        mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = kalman_map(x_hist)
        if has_one_sim:
            mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = mu_hist[0, ...], Sigma_hist[0, ...], mu_cond_hist[
                0, ...], Sigma_cond_hist[0, ...]
        return mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist
