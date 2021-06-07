# Jax implementation of a Linear Dynamical System
# Author:  Gerardo Duran-Martin (@gerdm)

import jax.numpy as jnp
from jax import random
from jax.ops import index_update
from jax.numpy.linalg import inv

class LinearDynamicalSystem:
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
    μ0: array(state_size)
        Mean of initial configuration
    Σ0: array(state_size, state_size) or 0
        Covariance of initial configuration. If value is set
        to zero, the initial state will be completely determined
        by μ0
    timesteps: int
        Total number of steps to sample
    """
    def __init__(self, A, C, Q, R, μ0, Σ0=None, timesteps=15):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.μ0 = μ0
        self.Σ0 = Σ0
        self.timesteps = timesteps
        self.state_size, _ = A.shape
        self.observation_size, _ = C.shape

    def sample(self, key, sample_intial_state=False):
        """
        # To-do: implement switching systems
        Simulate a run of (switching) stochastic linear dynamical systems

        Parameters
        ----------
        key: jax.random.PRNGKey
            Seed of initial random states

        Returns
        -------
        * array(timesteps, state_size):
            Simulation of Latent states
        * array(timesteps, observation_size):
            Simulation of observed states
        """
        key_z1, key_eps, key_delta = random.split(key, 3)
        if not sample_intial_state:
            z1 =self.μ0
        else:
            z1 = random.multivariate_normal(key_z1, self.μ0, self.Σ0)

        # Generate all future noise terms
        eps = random.multivariate_normal(key_eps, jnp.zeros(self.state_size), self.Q, (self.timesteps, ))
        delta = random.multivariate_normal(key_delta, jnp.zeros(self.observation_size), self.R, (self.timesteps, ))
        
        z_hist = jnp.zeros((self.timesteps, self.state_size))
        x_hist = jnp.zeros((self.timesteps, self.observation_size))

        z_hist = index_update(z_hist, 0, z1)
        x_hist = index_update(x_hist, 0, self.C @ z1 + delta[0])
        for t in range(1, self.timesteps):
            zt = self.A @ z_hist[t-1] + eps[t]
            xt = self.C @ zt + delta[t]

            z_hist = index_update(z_hist, t, zt)
            x_hist = index_update(x_hist, t, xt)
        
        return z_hist, x_hist

    def kalman_filter(self, x_hist):
        """
        Compute the online version of the Kalman-Filter, i.e,
        the one-step-ahead prediction for the hidden state or the
        time update step
        
        Parameters
        ----------
        x_hist: array(timesteps, observation_size)
        A: array(state_size, state_size)
            Transition matrix
        C: array(observation_size, state_size)
            Observation matrix
        Q: array(state_size, state_size)
            Transition covariance matrix
        R: array(observation_size, observation_size)
            Observation covariance
        μ0: array(state_size)
            Mean of initial configuration
        Σ0: array(state_size, state_size) or 0
            Covariance of initial configuration. If value is set
            to zero, the initial state will be completely determined
            by μ0
            
        Returns
        -------
        * array(timesteps, state_size):
            Filtered means μt
        * array(timesteps, state_size, state_size)
            Filtered covariances Σt
        * array(timesteps, state_size)
            Filtered conditional means μt|t-1
        * array(timesteps, state_size, state_size)
            Filtered conditional covariances Σt|t-1
        """
        I = jnp.eye(self.state_size)
        μ_hist = jnp.zeros((self.timesteps, self.state_size))
        Σ_hist = jnp.zeros((self.timesteps, self.state_size, self.state_size))
        Σ_cond_hist = jnp.zeros((self.timesteps, self.state_size, self.state_size))
        μ_cond_hist = jnp.zeros((self.timesteps, self.state_size))
        
        # Initial configuration
        K1 = self.Σ0 @ self.C.T @ inv(self.C @ self.Σ0 @ self.C.T + self.R)
        μ1 = self.μ0 + K1 @ (x_hist[0] - self.C @ self.μ0)
        Σ1 = (I - K1 @ self.C) @ self.Σ0

        μ_hist = index_update(μ_hist, 0, μ1)
        Σ_hist = index_update(Σ_hist, 0, Σ1)
        μ_cond_hist = index_update(μ_cond_hist, 0, self.μ0)
        Σ_cond_hist = index_update(Σ_hist, 0, self.Σ0)
        
        Σn = Σ1
        for n in range(1, self.timesteps):
            # Σn|{n-1}
            Σn_cond = self.A @ Σn @ self.A.T + self.Q
            St = self.C @ Σn_cond @ self.C.T + self.R
            Kn = Σn_cond @ self.C.T @ inv(St)

            # μn|{n-1} and xn|{n-1}
            μ_update = self.A @ μ_hist[n-1]
            x_update = self.C @ μ_update

            μn = μ_update + Kn @ (x_hist[n] - x_update)
            Σn = (I - Kn @ self.C) @ Σn_cond

            μ_hist = index_update(μ_hist, n, μn)
            Σ_hist = index_update(Σ_hist, n, Σn)
            μ_cond_hist = index_update(μ_cond_hist, n, μ_update)
            Σ_cond_hist = index_update(Σ_cond_hist, n, Σn_cond)
        
        return μ_hist, Σ_hist, μ_cond_hist, Σ_cond_hist

    def kalman_smoother(self, μ_hist, Σ_hist, μ_cond_hist, Σ_cond_hist):
        """
        Compute the offline version of the Kalman-Filter, i.e,
        the kalman smoother for the hidden state.
        Note that we require to independently run the kalman_filter function first
        
        Parameters
        ----------
        μ_hist: array(timesteps, state_size):
            Filtered means μt
        Σ_hist: array(timesteps, state_size, state_size)
            Filtered covariances Σt
        μ_cond_hist: array(timesteps, state_size)
            Filtered conditional means μt|t-1
        Σ_cond_hist: array(timesteps, state_size, state_size)
            Filtered conditional covariances Σt|t-1
            
        Returns
        -------
        * array(timesteps, state_size):
            Smoothed means μt
        * array(timesteps, state_size, state_size)
            Smoothed covariances Σt
        """
        timesteps, _ = μ_hist.shape
        state_size, _ = self.A.shape
        μ_hist_smooth = jnp.zeros((timesteps, state_size))
        Σ_hist_smooth = jnp.zeros((timesteps, state_size, state_size))

        μt_giv_T = μ_hist[-1, :]
        Σt_giv_T = Σ_hist[-1, :]

        # Update last step
        μ_hist_smooth = index_update(μ_hist_smooth, -1,  μt_giv_T)
        Σ_hist_smooth = index_update(Σ_hist_smooth, -1,  Σt_giv_T)

        elements = zip(μ_hist[-2::-1], Σ_hist[-2::-1, ...], μ_cond_hist[::-1, ...], Σ_cond_hist[::-1, ...])
        for t, (μtt, Σtt, μt_cond_next, Σt_cond_next) in enumerate(elements, 1):
            Jt  = Σtt @ self.A.T @ inv(Σt_cond_next)
            μt_giv_T = μtt + Jt @ (μt_giv_T - μt_cond_next)
            Σt_giv_T = Σtt + Jt @ (Σt_giv_T - Σt_cond_next) @ Jt.T
            
            μ_hist_smooth = index_update(μ_hist_smooth, -(t+1),  μt_giv_T)
            Σ_hist_smooth = index_update(Σ_hist_smooth, -(t+1), Σt_giv_T)
        
        return μ_hist_smooth, Σ_hist_smooth
