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

    @staticmethod
    def find_dishonest_intervals(z_hist):
        """
        Find the span of timesteps that the
        simulated systems turns to be in state 1
        
        Parameters
        ----------
        z_hist: array(n_samples)
            Result of running the system with two
            latent states
        
        Returns
        -------
        list of tuples with span of values
        """
        spans = []
        x_init = 0
        t = x_init
        for t, _ in enumerate(z_hist[:-1]):
            if z_hist[t+1] == 0 and z_hist[t] == 1:
                x_end = t
                spans.append((x_init, x_end))
            elif z_hist[t+1] == 1 and z_hist[t] == 0:
                x_init = t+1
        return spans
    
    def plot_inference(self, inference_values, z_hist, ax, state=1, map_estimate=False):
        """
        Plot the estimated smoothing/filtering/map of a sequence of hidden states.
        "Vertical gray bars denote times when the hidden
        state corresponded to state 1. Blue lines represent the
        posterior probability of being in that state given diﬀerent subsets
        of observed data." See Markov and Hidden Markov models section for more info
        
        Parameters
        ----------
        inference_values: array(n_samples, state_size)
            Result of runnig smoothing method
        z_hist: array(n_samples)
            Latent simulation
        ax: matplotlib.axes
        state: int
            Decide which state to highlight
        map_estimate: bool
            Whether to plot steps (simple plot if False)
        """
        n_samples = len(inference_values)
        xspan = np.arange(1, n_samples + 1)
        spans = self.find_dishonest_intervals(z_hist)
        if map_estimate:
            ax.step(xspan, inference_values, where="post")
        else:
            ax.plot(xspan, inference_values[:, state])

        for span in spans:
            ax.axvspan(*span, alpha=0.5, facecolor="tab:gray", edgecolor="none")
        ax.set_xlim(1, n_samples)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Roll number")
        
    def sample(self, n_samples, random_state=None):
        """
        Sample n_samples states of the discrete-variables HMM
        
        Parameters
        ----------
        n_samples: int
            Number of iterations in the process
        random_state: int or None
            random state of the system
        
        Returns
        -------
        * array(n_samples)
            History of latent states
        * array(n_samples)
            History of observed states
        """
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

    def _forward_step(self, x_hist):
        """
        Compute the "filtering" step of the HMM
        
        Parameters
        ----------
        x_hist: array(n_samples)
            History of observed states
            
        Returns
        -------
        * array(n_samples, state_size)
            Posterior over hidden states given the data seen
        """
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
        
        return α_hist, c_elements
    
    def _forward_backwards_step(self, x_hist):
        """
        Compute the intermediate smoothing step of the HMM.
        First compute the "filtering" terms and then the "forward" elements
        required to compute the smoothing elements
        
        Parameters
        ----------
        x_hist: array(n_samples)
            History of observed states
            
        Returns
        -------
        * array(n_samples, state_size)
            Posterior over hidden states given the data seen
        * array(n_samples, state_size)
            Intermediate elements to compute the smoothing term
        * array(n_samples)
            Coefficients 
        """
        n_samples = len(x_hist)
        α_hist, c_elements = self._forward_step(x_hist)
        β_next = np.ones(self.state_size)

        β_hist = np.zeros((n_samples, self.state_size))
        β_hist[-1] = β_next

        for n in range(2, n_samples + 1):
            β_next = (β_next * self.px[:, x_hist[-n+1]] * self.A).sum(axis=-1) / c_elements[-n+1]
            β_hist[-n] = β_next
        
        return α_hist, β_hist, c_elements
    
    def filter_smooth(self, x_hist):
        """
        Compute the "filtering" and "smoothing" steps of the HMM
        
        Parameters
        ----------
        x_hist: array(n_samples)
            History of observed states
            
        Returns
        -------
        Dictionary:
        * filtering: array(n_samples, state_size)
            Posterior over hidden states given the data seen
        * smoothing: array(n_samples, state_size)
            Posterior over hidden states conditional on all the data
        """ 
        α_hist, β_hist, _ = self._forward_backwards_step(x_hist)
        γ_hist = α_hist * β_hist
        
        return {
            "filtering": α_hist,
            "smoothing": γ_hist
        }
    
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
