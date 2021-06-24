# Library of nonlinear dynamical systems
# Author: Gerardo Durán-Martín (@gerdm)

import jax
import jax.numpy as jnp
from jax import random
from jax.ops import index_update
from math import ceil


class ExtendedKalmanFilter:
    """
    Implementation of the Extended Kalman Filter for a nonlinear
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
    
    def sample(self, key, x0, nsteps):
        """
        Sample discrete elements of a nonlinear system

        Parameters
        ----------
        key: jax.random.PRNGKey
        x0: array(state_size)
            Initial state of simulation
        nsteps: int
            Total number of steps to sample from the system

        Returns
        -------
        * array(nsamples, state_size)
            State-space values
        * array(nsamples, obs_size)
            Observed-space values
        """
        key, key_system_noise, key_obs_noise = random.split(key, 3)

        state_hist = jnp.zeros((nsteps, self.state_size))
        obs_hist = jnp.zeros((nsteps, self.obs_size))

        state_t = x0.copy()
        obs_t = self.fx(state_t)

        state_noise = random.multivariate_normal(key_system_noise, jnp.zeros((self.state_size,)), self.Q, (nsteps,))
        obs_noise = random.multivariate_normal(key_obs_noise, jnp.zeros((self.obs_size,)), self.R, (nsteps,))
        state_hist = index_update(state_hist, 0, state_t)
        obs_hist = index_update(obs_hist, 0, obs_t)

        for t in range(1, nsteps):
            state_t = self.fz(state_t) + state_noise[t]
            obs_t = self.fx(state_t) + obs_noise[t]

            state_hist = index_update(state_hist, t, state_t)
            obs_hist = index_update(obs_hist, t, obs_t)
        
        return state_hist, obs_hist


    def filter(self, init_state, sample_obs):
        """
        Run the Extended Kalman Filter algorithm over a set of observed samples.

        Parameters
        ----------
        init_state: array(state_size)
        sample_obs: array(nsamples, obs_size)


        Returns
        -------
        * array(nsamples, state_size)
            History of filtered mean terms
        * array(nsamples, state_size, state_size)
            History of filtered covariance terms
        """
        I = jnp.eye(self.state_size)
        nsamples = len(sample_obs)
        Vt = self.Q.copy()

        #mu_t = hidden_states[0]
        mu_t = init_state

        mu_hist = jnp.zeros((nsamples, self.state_size))
        V_hist = jnp.zeros((nsamples, self.state_size, self.state_size))

        mu_hist = index_update(mu_hist, 0, mu_t)
        V_hist = index_update(V_hist, 0, Vt)

        for t in range(1, nsamples):
            Gt = self.Dfz(mu_t)
            mu_t_cond = self.fz(mu_t)
            Vt_cond = Gt @ Vt @ Gt + self.Q
            Ht = self.Dfx(self.fx(mu_t_cond))

            Kt = Vt_cond @ Ht.T @ jnp.linalg.inv(Ht @ Vt_cond @ Ht.T + self.R)
            mu_t = mu_t_cond + Kt @ (sample_obs[t] - self.fx(mu_t_cond))
            Vt = (I - Kt @ Ht) @ Vt_cond

            mu_hist = index_update(mu_hist, t, mu_t)
            V_hist = index_update(V_hist, t, Vt)
        
        return mu_hist, V_hist


class ContinuousExtendedKalmanFilter:
    """
    Extended Kalman Filter for a nonlinear continuous time
    dynamical system with observations in discrete time.
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
    
    def sample(self, key, x0, T, nsamples, dt=0.01, noisy=False):
        """
        Run the Extended Kalman Filter algorithm. First, we integrate
        up to time T, then we obtain nsamples equally-spaced points. Finally,
        we transform the latent space to obtain the observations

        Parameters
        ----------
        key: jax.random.PRNGKey
            Initial seed
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
        Run the Extended Kalman Filter algorithm over a set of observed samples.

        Parameters
        ----------
        sample_state: array(nsamples, state_size)
        sample_obs: array(nsamples, obs_size)
        jump_size: int
        dt: float

        Returns
        -------
        * array(nsamples, state_size)
            History of filtered mean terms
        * array(nsamples, state_size, state_size)
            History of filtered covariance terms
        """
        I = jnp.eye(self.state_size)
        nsamples = len(sample_state)
        Vt = self.R.copy()
        mu_t = sample_state[0]

        mu_hist = jnp.zeros((nsamples, self.state_size))
        V_hist = jnp.zeros((nsamples, self.state_size, self.state_size))

        mu_hist = index_update(mu_hist, 0, mu_t)
        V_hist = index_update(V_hist, 0, Vt)

        for t in range(1, nsamples):
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
            mu_t = mu_t_cond + Kt @ (sample_obs[t] - self.fx(mu_t_cond))
            Vt = (I - Kt @ Ht) @ Vt_cond

            mu_hist = index_update(mu_hist, t, mu_t)
            V_hist = index_update(V_hist, t, Vt)
        
        return mu_hist, V_hist


class UnscentedKalmanFilter:
    """
    Implementation of the Unscented Kalman Filter for discrete time systems
    """
    def __init__(self, fz, fx, Q, R, alpha, beta, kappa):
        self.fz = fz
        self.fx = fx
        self.Q = Q
        self.R = R
        self.d, _ = Q.shape
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = alpha ** 2 * (self.d + kappa) - self.d
        self.gamma = jnp.sqrt(self.d + self.lmbda)
    
    @staticmethod
    def sqrtm(M):
        """
        Compute the matrix square-root of a hermitian
        matrix M. i,e, R such that RR = M
        
        Parameters
        ----------
        M: array(m, m)
            Hermitian matrix
        
        Returns
        -------
        array(m, m): square-root matrix
        """
        evals, evecs = jnp.linalg.eigh(M)
        R = evecs @ jnp.sqrt(jnp.diag(evals)) @ jnp.linalg.inv(evecs)
        return R
    
    def sample(self, key, x0, nsteps):
        """
        Sample discrete elements of a nonlinear system

        Parameters
        ----------
        key: jax.random.PRNGKey
        x0: array(state_size)
            Initial state of simulation
        nsteps: int
            Total number of steps to sample from the system

        Returns
        -------
        * array(nsamples, state_size)
            State-space values
        * array(nsamples, obs_size)
            Observed-space values
        """
        key, key_system_noise, key_obs_noise = random.split(key, 3)

        state_hist = jnp.zeros((nsteps, self.d))
        obs_hist = jnp.zeros((nsteps, self.d))

        state_t = x0.copy()
        obs_t = self.fx(state_t)

        state_noise = random.multivariate_normal(key_system_noise, jnp.zeros((self.d,)), self.Q, (nsteps,))
        obs_noise = random.multivariate_normal(key_obs_noise, jnp.zeros((self.d,)), self.R, (nsteps,))
        state_hist = index_update(state_hist, 0, state_t)
        obs_hist = index_update(obs_hist, 0, obs_t)

        for t in range(1, nsteps):
            state_t = self.fz(state_t) + state_noise[t]
            obs_t = self.fx(state_t) + obs_noise[t]

            state_hist = index_update(state_hist, t, state_t)
            obs_hist = index_update(obs_hist, t, obs_t)
        
        return state_hist, obs_hist

    def filter(self, init_state, sample_obs):
        """
        Run the Unscented Kalman Filter algorithm over a set of observed samples.

        Parameters
        ----------
        sample_obs: array(nsamples, obs_size)

        Returns
        -------
        * array(nsamples, state_size)
            History of filtered mean terms
        * array(nsamples, state_size, state_size)
            History of filtered covariance terms
        """
        wm_vec = jnp.array([1 / (2 * (self.d + self.lmbda)) if i > 0
                            else self.lmbda / (self.d + self.lmbda)
                            for i in range(2 * self.d + 1)])
        wc_vec = jnp.array([1 / (2 * (self.d + self.lmbda)) if i > 0
                            else self.lmbda / (self.d + self.lmbda) + (1 - self.alpha ** 2 + self.beta)
                            for i in range(2 * self.d + 1)])
        nsteps, _ = sample_obs.shape
        #mu_t = sample_obs[0]
        mu_t = init_state
        Sigma_t = self.Q

        mu_hist = jnp.zeros((nsteps, self.d))
        Sigma_hist = jnp.zeros((nsteps, self.d, self.d))

        mu_hist = index_update(mu_hist, 0, mu_t)
        Sigma_hist = index_update(Sigma_hist, 0, Sigma_t)

        for t in range(1, nsteps):
            # TO-DO: use jax.scipy.linalg.sqrtm when it gets added to lib
            comp1 = mu_t[:, None] + self.gamma * self.sqrtm(Sigma_t)
            comp2 = mu_t[:, None] - self.gamma * self.sqrtm(Sigma_t)
            #print('kpm')
            #print([mu_t.shape, comp1.shape, comp2.shape])
            #sigma_points = jnp.c_[mu_t, comp1, comp2]
            sigma_points = jnp.concatenate((mu_t[:, None], comp1, comp2), axis=1)

            z_bar = self.fz(sigma_points)
            mu_bar = z_bar @ wm_vec
            Sigma_bar = (z_bar - mu_bar[:, None])
            Sigma_bar = jnp.einsum("i,ji,ki->jk", wc_vec, Sigma_bar, Sigma_bar) + self.Q

            comp1 = mu_bar[:, None] + self.gamma * self.sqrtm(self.Q)
            comp2 = mu_bar[:, None] - self.gamma * self.sqrtm(self.Q)
            #sigma_points = jnp.c_[mu_bar, comp1, comp2]
            sigma_points = jnp.concatenate((mu_bar[:, None], comp1, comp2), axis=1)

            x_bar = self.fx(z_bar)
            x_hat = x_bar @ wm_vec
            St = (x_bar - x_hat[:, None])
            St = jnp.einsum("i,ji,ki->jk", wc_vec, St, St) + self.R

            mu_hat_component = (z_bar - mu_bar[:, None])
            x_hat_component = (x_bar - x_hat[:, None])
            Sigma_bar_y = jnp.einsum("i,ji,ki->jk", wc_vec, mu_hat_component, x_hat_component)
            Kt = Sigma_bar_y @ jnp.linalg.inv(St)

            mu_t = mu_bar + Kt @ (sample_obs[t] - x_hat)
            Sigma_t = Sigma_bar - Kt @ St @ Kt.T
            
            mu_hist = index_update(mu_hist, t, mu_t)
            Sigma_hist = index_update(Sigma_hist, t, Sigma_t)

        return mu_hist, Sigma_hist
