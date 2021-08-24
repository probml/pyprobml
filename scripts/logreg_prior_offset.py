# We illustrate effect of prior on parameters for logistic regression
# Based on fig 11.3 of
# [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/).

import superimport

import pyprobml_utils as pml
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from scipy import stats
import arviz as az
from scipy.special import expit

sigmas = [1.5, 10]
fig, ax = plt.subplots()
colors = ['r', 'k']
np.random.seed(0)
for i in range(2):
  sigma = sigmas[i]
  N = 1000
  a = stats.norm(0, sigma).rvs((N))
  logits = a
  probs = expit(logits)
  label = r'variance={:0.2f}'.format(sigma)
  az.plot_kde(probs, ax=ax, plot_kwargs = {'color': colors[i]}, label=label, legend=True)
pml.savefig('logreg_prior_offset.pdf', dpi=300)
plt.show()