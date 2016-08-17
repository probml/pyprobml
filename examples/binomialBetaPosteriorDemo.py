#!/usr/bin/env python

# Plots Beta-Binomial distribution along with the prior and likelihood.

import matplotlib.pyplot as pl
import numpy as np
import scipy
from scipy.stats import beta

alphas = [2, 2, 1, 1]
betas = [2, 2, 1, 1]
Ns = [4, 40, 4, 40]
ks = [1, 10, 1, 10]
plots = ['betaPostInfSmallSample', 'betaPostInfLargeSample',
         'betaPostUninfSmallSample', 'betaPostUninfLargeSample']

x = np.linspace(0.001, 0.999, 50)
for i in range(len(plots)):
  alpha_prior = alphas[i]
  beta_prior = betas[i]
  N = Ns[i]
  k = ks[i]
  alpha_post = alpha_prior + N - k
  beta_post = beta_prior + k
  alpha_lik = N - k + 1
  beta_lik = k + 1

  pl.plot(x, beta.pdf(x, alpha_prior, beta_prior), 'r-', 
          label='prior Be(%2.1f, %2.1f)' % (alpha_prior, beta_prior))
  pl.plot(x, beta.pdf(x, alpha_lik, beta_lik), 'k:', 
          label='lik Be(%2.1f, %2.1f)' % (alpha_lik, beta_lik))
  pl.plot(x, beta.pdf(x, alpha_post, beta_post), 'b-', 
          label='post Be(%2.1f, %2.1f)' % (alpha_post, beta_post))
  pl.legend(loc='upper left')
  pl.savefig(plots[i] + '.png')
  pl.show()
