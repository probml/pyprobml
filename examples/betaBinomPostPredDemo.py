#!/usr/bin/env python3
#
# Plots the posterior and plugin predictives for the Beta-Binomial distribution.

from scipy.misc import comb
from scipy.special import beta
from scipy.stats import binom
import matplotlib.pyplot as pl
import numpy as np
from utils.util import save_fig

N = 10
# Hyperparameters
a = 2;
b = 2;
N1 = 4
N0 = 1

ind = np.arange(N+1)
post_a = a + N1
post_b = b + N0

distribution = []
for k in range(N+1):
  distribution.append(comb(N,k) * beta(k+post_a, N-k+post_b) / beta(post_a, post_b))

fig,ax = pl.subplots()
rects = ax.bar(ind, distribution, align='center')
ax.set_title('posterior predictive')
ax.set_xticks(list(range(N+1)))
ax.set_xticklabels(list(range(N+1)))
save_fig('BBposteriorpred.png')
pl.show()

mu = (post_a - 1) / float(post_a + post_b - 2)
distribution = []
rv = binom(N, mu)
for k in range(N+1):
  distribution.append(rv.pmf(k))

fig,ax = pl.subplots()
rects = ax.bar(ind, distribution, align='center')
ax.set_title('incremental predictive')
ax.set_xticks(list(range(N+1)))
ax.set_xticklabels(list(range(N+1)))

save_fig('BBpluginpred.pdf')
pl.show()


