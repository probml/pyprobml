# Implementation of the Hidden Markov Model for discrete observations
# Author: Gerardo Duran-Martin (@gerdm)

import numpy as np
from numpy.random import seed, choice


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

    def sample(self, n_samples, random_state=None):
        seed(random_state)
        latent_states = np.arange(self.state_size)
        obs_states = np.arange(self.observation_size)

        z_hist = np.zeros(n_samples, dtype=int)
        x_hist = np.zeros(n_samples, dtype=int)

        zt = choice(latent_states, p=self.pi)
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
        alpha_hist = np.zeros((n_samples, self.state_size))
        c_elements = np.zeros(n_samples) # normalization constants

        alpha_n = self.pi * self.px[:, x_hist[0]]
        cn = alpha_n.sum()
        alpha_n = alpha_n / cn

        alpha_hist[0] = alpha_n
        c_elements[0] = cn

        for t in range(1, n_samples):
            alpha_n = self.px[:, x_hist[t]] * (alpha_n[:, None] * self.A).sum(axis=0)
            cn = alpha_n.sum()
            alpha_n = alpha_n / cn

            alpha_hist[t] = alpha_n
            c_elements[t] = cn

        return alpha_hist, np.sum(np.log(c_elements))

    def backwards_filtering(self, x_hist):
        n_samples = len(x_hist)
        beta_next = np.ones(self.state_size)

        beta_hist = np.zeros((n_samples, self.state_size))
        beta_hist[-1] = beta_next

        for t in range(2, n_samples + 1):
            beta_next = (beta_next * self.px[:, x_hist[-t + 1]] * self.A).sum(axis=1)
            beta_next = beta_next / beta_next.sum()
            beta_hist[-t] = beta_next
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
        logp_hist = np.zeros((n_samples, self.state_size))
        wn = np.log(self.A) + np.log(self.pi) + np.log(self.px[:, x_hist[0]])
        wn = wn.max(axis=1)
        logp_hist[0] = wn

        for t in range(1, n_samples):
            wn = np.log(self.A) + np.log(self.px[:, x_hist[t]]) + wn
            wn = wn.max(axis=1)
            logp_hist[t] = wn

        return logp_hist.argmax(axis=1)