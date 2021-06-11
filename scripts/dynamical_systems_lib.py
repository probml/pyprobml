# Library of continous-time nonlinear dynamical systems
# Author: Gerardo Durán-Martín (@gerdm)

import jax
import jax.numpy as jnp
from jax import random
from jax.ops import index_update

class ExtendedKalmanFilter:
    """
    Implementation of the Extended Kalman Filter for a nonlinear-continous
    dynamical system with discrete observations
    """
    def __init__(self, fz, fx, Q, R):
        self.fz = fz
        self.fx = fx
        self.Dfz = jax.jacfwd(fz)
        self.Dfx = jax.jacfwd(fx)
        self.Q = Q
        self.R = R
        self.state_size, _ = Q.shape
        self.obs_size, _ = R.shape
        
    @staticmethod
    def _rk2(x0, f, nsteps, dt):
        """
        class-independent second-order Runge-Kutta method
        
        Parameters
        ----------
        x0: array(state_size, )
            Initial state of the system
        f: function
            Function to integrate. Must return jax.numpy
            array of size state_size
        nsteps: int
            Total number of steps to integrate
        dt: float
            integration step size
        
        Returns
        -------
        array(nsteps, state_size)
            Integration history
        """
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
    
    def simulate(self, x0, key, T, n_samples, dt=0.01, noisy=False):
        """
        Run the Extended Kalman Filter algorithm. First, we integrate
        up to time T, then we obtain n_samples equally-spaced points. Finally,
        we transform the latent space to obtain the observations

        Parameters
        ----------
        x0: array(state_size)
            Initial state of simulation
        key: jax.random.PRNGKey
        T: float
            Final time of integration
        n_samples: int
            Number of observations to take from the total integration
        dt: float
            integration step size
        noisy: bool
            Whether to (naively) add noise to the state space

        Returns
        -------
        * array(n_samples, state_size)
            State-space values
        * array(n_samples, obs_size)
            Observed-space values
        * int
            Number of observations skipped between one
            datapoint and the next
        """
        nsteps = int(T // dt)
        jump_size = nsteps // n_samples
        key_state, key_obs = random.split(key)
        state_noise = random.multivariate_normal(key_state, jnp.zeros(self.state_size), self.Q, (nsteps,))
        obs_noise = random.multivariate_normal(key_obs, jnp.zeros(self.obs_size), self.R, (nsteps,)) 
        simulation = self._rk2(x0, self.fz, nsteps, dt)
        
        if noisy:
            simulation = simulation + jnp.sqrt(dt) * state_noise
        
        sample_state = simulation[::jump_size]
        sample_obs = jnp.apply_along_axis(self.fx, 1, sample_state) + obs_noise[:len(sample_state)]
        
        return sample_state, sample_obs, jump_size
    
    def _Vt_dot(self, V, G):
        return G @ V @ G + self.Q
    
    def estimate(self, sample_state, sample_obs, jump_size, dt):
        """
        Run the Extended Kalman Filter algorithm over a set of samples
        obtained using the `simulate` method

        Parameters
        ----------
        sample_state: array(n_samples, state_size)
        sample_obs: array(n_samples, obs_size)
        jump_size: int
        dt: float

        Returns
        -------
        * array(n_samples, state_size)
            History of filtered mean terms
        * array(n_samples, state_size, state_size)
            History of filtered covariance terms
        """
        I = jnp.eye(self.state_size)
        n_samples = len(sample_state)
        Vt = self.R.copy()
        mu_t = sample_state[0]

        mu_hist = jnp.zeros((n_samples, self.state_size))
        V_hist = jnp.zeros((n_samples, self.state_size, self.state_size))

        mu_hist = index_update(mu_hist, 0, mu_t)
        V_hist = index_update(V_hist, 0, Vt)

        for t in range(1, n_samples):
            for _ in range(jump_size):
                k1 = self.fz(mu_t)
                k2 = self.fz(mu_t + dt * k1)
                mu_t = mu_t + dt * (k1 + k2) / 2

                Gt = self.Dfz(mu_t)
                k1 = self._Vt_dot(Vt, Gt)
                k2 = self._Vt_dot(Vt + dt * k1, Gt)
                Vt = Vt + dt * (k1 + k2) / 2
            
            mu_t_cond = mu_t
            Vt_cond = Vt
            Ht = self.Dfx(mu_t_cond)

            Kt = Vt_cond @ Ht.T @ jnp.linalg.inv(Ht @ Vt_cond @ Ht.T + self.R)
            mu_t = mu_t_cond + Kt @ (sample_obs[t] - mu_t_cond)
            Vt = (I - Kt @ Ht) @ Vt_cond

            mu_hist = index_update(mu_hist, t, mu_t)
            V_hist = index_update(V_hist, t, Vt)
        
        return mu_hist, V_hist
