# Illustration of possible behaviors of variational EM
# Author : Aleyna Kara
# This file is generated from varEMbound.m

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

def logist(x, k):
  return 1 / (1 + np.exp(-k * x))

def logist_gauss(x, k, mu, sigma):
  return 1 / (1 + np.exp(-k*x)) + (1 / (2*np.pi*sigma))* np.exp(-(x - mu)**2 / (sigma**2))

def plot_var_em_bound(x, true_log_ll, lower_bound, title):

  plt.plot(x, true_log_ll, '-b', linewidth=3)
  plt.text(2.5, y1[-1] + 0.02, 'true log-likelihood',  fontweight='bold')

  plt.plot(x, lower_bound, ':r', linewidth=3)
  plt.text(2.8, 0.9,  'lower bound',  fontweight='bold')

  plt.xlim([0, 4])
  plt.ylim([0.5, np.max(y1) + 0.05])
  plt.xlabel('training time',  fontweight='bold')
  plt.xticks([])
  plt.yticks([])
  pml.savefig(f'{title}.pdf', dpi=300)
  plt.show()

n = 41
x = np.linspace(0, 4, n)
k, mu, sigma = 2, 1, 0.9

y1 = logist(x, k)
lower_bound = logist(x, 1)

plot_var_em_bound(x, y1, lower_bound, 'varEMbound1')

y1 = logist_gauss(x, k, mu, sigma)

plot_var_em_bound(x, y1, lower_bound, 'varEMbound2')