
# Plots the posterior and plugin predictives for the Beta-Binomial distribution.


import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

from scipy.special import comb, beta
from scipy.stats import binom


N = 10 # Future sample size M
# Hyperparameters
a = 1
b = 1
N1 = 4
N0 = 1

ind = np.arange(N+1)
post_a = a + N1
post_b = b + N0

# Compound beta-binomial distribution
distribution = []
for k in range(N+1):
  distribution.append(comb(N,k) * beta(k+post_a, N-k+post_b) / beta(post_a, post_b))

fig,ax = plt.subplots()
rects = ax.bar(ind, distribution, align='center')
ax.set_title('posterior predictive')
ax.set_xticks(list(range(N+1)))
ax.set_xticklabels(list(range(N+1)))
pml.savefig('BBpostpred.pdf')
plt.show()

# Plugin binomial distribution
mu = (post_a - 1) / float(post_a + post_b - 2) # MAP estimate
distribution = []
rv = binom(N, mu)
for k in range(N+1):
  distribution.append(rv.pmf(k))

fig,ax = plt.subplots()
rects = ax.bar(ind, distribution, align='center')
ax.set_title('plugin predictive')
ax.set_xticks(list(range(N+1)))
ax.set_xticklabels(list(range(N+1)))
pml.savefig('BBpluginpred.pdf')
plt.show()


