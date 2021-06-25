# Implementation of the Hidden Markov Model for discrete observations
# Author: Gerardo Duran-Martin (@gerdm)

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed, choice

class HMMDiscrete:
    def __init__(self, A, px, π):
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
        π: array(state_size)
            Array of initial-state probabilities
        """
        self.A = A
        self.px = px
        self.π = π
        self.state_size, self.observation_size = px.shape

    def sample(self, n_samples, random_state=None):
        seed(random_state)
        latent_states = np.arange(self.state_size)
        obs_states = np.arange(self.observation_size)
        
        z_hist = np.zeros(n_samples, dtype=int)
        x_hist = np.zeros(n_samples, dtype=int)
        
        zt = choice(latent_states, p=self.π)
        xt = choice(obs_states, p=self.px[zt])
        z_hist[0] = zt
        x_hist[0] = xt
        
        for t in range(1, n_samples):
            zt = choice(latent_states, p=self.A[zt])
            xt = choice(obs_states, p=self.px[zt])
            z_hist[t] = zt
            x_hist[t] = xt
        return z_hist, x_hist

    def forwards(self, x_hist):
        n_samples = len(x_hist)
        α_hist = np.zeros((n_samples, self.state_size))
        c_elements = np.zeros(n_samples)
        
        αn = self.π * self.px[:, x_hist[0]]
        cn = αn.sum()
        αn = αn / cn

        α_hist[0] = αn
        c_elements[0] = cn

        for n in range(1, n_samples):
            αn = self.px[:, x_hist[n]] * (αn[:, None] * self.A).sum(axis=0)
            cn = αn.sum()
            αn = αn / cn

            α_hist[n] = αn
            c_elements[n] = cn

        loglik = np.sum(np.log(c_elements))
        return α_hist, c_elements
    
    def forwards_backwards(self, x_hist):
        n_samples = len(x_hist)
        α_hist, c_elements = self.forwards(x_hist)
        β_next = np.ones(self.state_size)

        β_hist = np.zeros((n_samples, self.state_size))
        β_hist[-1] = β_next

        for n in range(2, n_samples + 1):
            β_next = (β_next * self.px[:, x_hist[-n+1]] * self.A).sum(axis=-1) / c_elements[-n+1]
            β_hist[-n] = β_next

        γ_hist = α_hist * β_hist
        return γ_hist

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
        logp_hist = np.zeros((n_samples, self.state_size))
        wn = np.log(self.A) + np.log(self.π) + np.log(self.px[:, x_hist[0]])
        wn = wn.max(axis=1)
        logp_hist[0] = wn

        for n in range(1, n_samples):
            wn = np.log(self.A) + np.log(self.px[:, x_hist[n]]) + wn
            wn = wn.max(axis=1)
            logp_hist[n] = wn
            
        return logp_hist.argmax(axis=1)    
