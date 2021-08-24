# Implementation of the Kalman Filter for 
# continuous time series

# Author: Gerardo Durán-Martín (@gerdm)

import superimport

import jax
import jax.numpy as jnp
from math import ceil
from jax import random
from jax.ops import index_update
from jax.numpy.linalg import inv

class ContinuousKalmanFilter:
    """
    Implementation of the Kalman Filter procedure of a
    continuous linear dynamical system with discrete observations

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
    """
    def __init__(self, A, C, Q, R, mu0, Sigma0):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        self.obs_size, self.state_size = C.shape

    @staticmethod
    def _rk2(x0, M, nsteps, dt):
        """
        class-independent second-order Runge-Kutta method for linear systems
        
        Parameters
        ----------
        x0: array(state_size, )
            Initial state of the system
        M: array(state_size, K)
            Evolution matrix
        nsteps: int
            Total number of steps to integrate
        dt: float
            integration step size
        
        Returns
        -------
        array(nsteps, state_size)
            Integration history
        """
        def f(x): return M @ x
        input_dim, *_ = x0.shape
        simulation = jnp.zeros((nsteps, input_dim))
        simulation = index_update(simulation, 0, x0)
        
        xt = x0.copy()
        for t in range(1, nsteps):
            k1 = f(xt)
            k2 = f(xt + dt * k1)
            xt = xt + dt * (k1 + k2) / 2
            simulation = index_update(simulation, t, xt)
        return simulation

    def sample(self, key, x0, T, nsamples, dt=0.01, noisy=False):
        """
        Run the Kalman Filter algorithm. First, we integrate
        up to time T, then we obtain nsamples equally-spaced points. Finally,
        we transform the latent space to obtain the observations

        Parameters
        ----------
        key: jax.random.PRNGKey
        x0: array(state_size)
            Initial state of simulation
        T: float
            Final time of integration
        nsamples: int
            Number of observations to take from the total integration
        dt: float
            integration step size
        noisy: bool
            Whether to (naively) add noise to the state space

        Returns
        -------
        * array(nsamples, state_size)
            State-space values
        * array(nsamples, obs_size)
            Observed-space values
        * int
            Number of observations skipped between one
            datapoint and the next
        """
        nsteps = ceil(T / dt)
        jump_size = ceil(nsteps / nsamples)
        correction = nsamples - ceil(nsteps / jump_size)
        nsteps += correction * jump_size

        key_state, key_obs = random.split(key)
        state_noise = random.multivariate_normal(key_state, jnp.zeros(self.state_size), self.Q, (nsteps,))
        obs_noise = random.multivariate_normal(key_obs, jnp.zeros(self.obs_size), self.R, (nsteps,)) 
        simulation = self._rk2(x0, self.A, nsteps, dt)
        
        if noisy:
            simulation = simulation + state_noise
        
        sample_state = simulation[::jump_size]
        sample_obs = jnp.einsum("ij,si->si", self.C, sample_state) + obs_noise[:len(sample_state)]
        
        return sample_state, sample_obs, jump_size

    def filter(self, x_hist, jump_size, dt):
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
        I = jnp.eye(self.state_size)
        timesteps, *_ = x_hist.shape
        mu_hist = jnp.zeros((timesteps, self.state_size))
        Sigma_hist = jnp.zeros((timesteps, self.state_size, self.state_size))
        Sigma_cond_hist = jnp.zeros((timesteps, self.state_size, self.state_size))
        mu_cond_hist = jnp.zeros((timesteps, self.state_size))
        
        # Initial configuration
        K1 = self.Sigma0 @ self.C.T @ inv(self.C @ self.Sigma0 @ self.C.T + self.R)
        mu1 = self.mu0 + K1 @ (x_hist[0] - self.C @ self.mu0)
        Sigma1 = (I - K1 @ self.C) @ self.Sigma0

        mu_hist = index_update(mu_hist, 0, mu1)
        Sigma_hist = index_update(Sigma_hist, 0, Sigma1)
        mu_cond_hist = index_update(mu_cond_hist, 0, self.mu0)
        Sigma_cond_hist = index_update(Sigma_hist, 0, self.Sigma0)
        
        Sigman = Sigma1.copy()
        mun = mu1.copy()
        for n in range(1, timesteps):
            # Runge-kutta integration step
            for _ in range(jump_size):
                k1 = self.A @ mun
                k2 = self.A @ (mun + dt * k1)
                mun = mun + dt * (k1 + k2) / 2

                k1 = self.A @ Sigman @ self.A.T + self.Q
                k2 = self.A @ (Sigman + dt * k1) @ self.A.T + self.Q
                Sigman = Sigman + dt * (k1 + k2) / 2

            Sigman_cond = Sigman.copy()
            St = self.C @ Sigman_cond @ self.C.T + self.R
            Kn = Sigman_cond @ self.C.T @ inv(St)

            mu_update = mun.copy()
            x_update = self.C @ mun
            mun = mu_update + Kn @ (x_hist[n] - x_update)
            Sigman = (I - Kn @ self.C) @ Sigman_cond

            mu_hist = index_update(mu_hist, n, mun)
            Sigma_hist = index_update(Sigma_hist, n, Sigman)
            mu_cond_hist = index_update(mu_cond_hist, n, mu_update)
            Sigma_cond_hist = index_update(Sigma_cond_hist, n, Sigman_cond)
        
        return mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist
