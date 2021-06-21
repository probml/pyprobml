# Jax implementation of a Linear Dynamical System
# Author:  Gerardo Durán-Martín (@gerdm)

import jax
import jax.numpy as jnp
from math import ceil
from jax import random
from jax.ops import index_update
from jax.numpy.linalg import inv

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
        self.C = C
        self.Q = Q
        self.R = R
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        self.timesteps = timesteps
        self.state_size, _ = A.shape
        self.observation_size, _ = C.shape

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
        key_z1, key_system_noise, key_obs_noise = random.split(key, 3)
        if not sample_intial_state:
            state_t =self.mu0 * jnp.ones((n_samples, self.state_size))
        else:
            state_t = random.multivariate_normal(key_z1, self.mu0, self.Sigma0, (n_samples,))

        # Generate all future noise terms
        zeros_state = jnp.zeros(self.state_size)
        zeros_obs = jnp.zeros(self.observation_size)
        system_noise = random.multivariate_normal(key_system_noise, zeros_state, self.Q, (n_samples, self.timesteps))
        obs_noise = random.multivariate_normal(key_obs_noise, zeros_obs, self.R, (n_samples, self.timesteps))
        
        state_hist = jnp.zeros((n_samples, self.timesteps, self.state_size))
        obs_hist = jnp.zeros((n_samples, self.timesteps, self.observation_size))

        obs_t = jnp.einsum("ij,sj->si", self.C, state_t) + obs_noise[:, 0, :]

        state_hist = index_update(state_hist, jax.ops.index[:, 0, :], state_t)
        obs_hist = index_update(obs_hist, jax.ops.index[:, 0, :], obs_t)

        for t in range(1, self.timesteps):
            system_noise_t = system_noise[:, t, :]
            obs_noise_t = obs_noise[:, t, :]

            state_new = jnp.einsum("ij,sj->si", self.A, state_t) + system_noise_t
            obs_t = jnp.einsum("ij,sj->si", self.C, state_new) + obs_noise_t
            state_t = state_new

            state_hist = index_update(state_hist, jax.ops.index[:, t, :], state_t)
            obs_hist = index_update(obs_hist, jax.ops.index[:, t, :], obs_t)

        if n_samples == 1:
            state_hist = state_hist[0, ...]
            obs_hist = obs_hist[0, ...]
        return state_hist, obs_hist

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
        I = jnp.eye(self.state_size)
        mu_hist = jnp.zeros((self.timesteps, self.state_size))
        Sigma_hist = jnp.zeros((self.timesteps, self.state_size, self.state_size))
        Sigma_cond_hist = jnp.zeros((self.timesteps, self.state_size, self.state_size))
        mu_cond_hist = jnp.zeros((self.timesteps, self.state_size))
        
        # Initial configuration
        K1 = self.Sigma0 @ self.C.T @ inv(self.C @ self.Sigma0 @ self.C.T + self.R)
        mu1 = self.mu0 + K1 @ (x_hist[0] - self.C @ self.mu0)
        Sigma1 = (I - K1 @ self.C) @ self.Sigma0

        mu_hist = index_update(mu_hist, 0, mu1)
        Sigma_hist = index_update(Sigma_hist, 0, Sigma1)
        mu_cond_hist = index_update(mu_cond_hist, 0, self.mu0)
        Sigma_cond_hist = index_update(Sigma_hist, 0, self.Sigma0)
        
        Sigman = Sigma1
        for n in range(1, self.timesteps):
            # Sigman|{n-1}
            Sigman_cond = self.A @ Sigman @ self.A.T + self.Q
            St = self.C @ Sigman_cond @ self.C.T + self.R
            Kn = Sigman_cond @ self.C.T @ inv(St)

            # mun|{n-1} and xn|{n-1}
            mu_update = self.A @ mu_hist[n-1]
            x_update = self.C @ mu_update

            mun = mu_update + Kn @ (x_hist[n] - x_update)
            Sigman = (I - Kn @ self.C) @ Sigman_cond

            mu_hist = index_update(mu_hist, n, mun)
            Sigma_hist = index_update(Sigma_hist, n, Sigman)
            mu_cond_hist = index_update(mu_cond_hist, n, mu_update)
            Sigma_cond_hist = index_update(Sigma_cond_hist, n, Sigman_cond)
        
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

        # Update last step
        mu_hist_smooth = index_update(mu_hist_smooth, -1,  mut_giv_T)
        Sigma_hist_smooth = index_update(Sigma_hist_smooth, -1,  Sigmat_giv_T)

        elements = zip(mu_hist[-2::-1], Sigma_hist[-2::-1, ...], mu_cond_hist[::-1, ...], Sigma_cond_hist[::-1, ...])
        for t, (mutt, Sigmatt, mut_cond_next, Sigmat_cond_next) in enumerate(elements, 1):
            Jt  = Sigmatt @ self.A.T @ inv(Sigmat_cond_next)
            mut_giv_T = mutt + Jt @ (mut_giv_T - mut_cond_next)
            Sigmat_giv_T = Sigmatt + Jt @ (Sigmat_giv_T - Sigmat_cond_next) @ Jt.T
            
            mu_hist_smooth = index_update(mu_hist_smooth, -(t+1),  mut_giv_T)
            Sigma_hist_smooth = index_update(Sigma_hist_smooth, -(t+1), Sigmat_giv_T)
        
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
            mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = mu_hist[0, ...], Sigma_hist[0, ...], mu_cond_hist[0, ...], Sigma_cond_hist[0, ...]
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
            mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = mu_hist[None, ...], Sigma_hist[None, ...], mu_cond_hist[None, ...], Sigma_cond_hist[None, ...]
            has_one_sim = True
        smoother_map = jax.vmap(self.__kalman_smoother, 0)
        mu_hist_smooth, Sigma_hist_smooth = smoother_map(mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist)
        if has_one_sim:
            mu_hist_smooth, Sigma_hist_smooth = mu_hist_smooth[0, ...], Sigma_hist_smooth[0, ...]
        return mu_hist_smooth, Sigma_hist_smooth

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