# -*- coding: utf-8 -*-
# Author: Ang Ming Liang

import superimport

import numpy as np
import matplotlib.pyplot as plt
from arspy.ars import adaptive_rejection_sampling

a , b = -2, 0
domain = [-float('inf'), 0]
n_samples = 20000
sigma = 3 

def halfgaussian_logpdf(x):
  out = np.log(np.exp(-x**2/sigma))*np.heaviside(-x,1)
  return out

xs = np.arange(-3*sigma, 3*sigma, 0.1)
y = np.exp(halfgaussian_logpdf(xs))

samples = adaptive_rejection_sampling(logpdf=halfgaussian_logpdf, a=a, b=b, domain=domain, n_samples=n_samples)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))

# Title
ax1.set_title("f(x) half-guassian")

# Fix the plot size
ax1.set_xlim(-3*sigma, 3*sigma)
ax1.set_ylim(0,1)

ax1.plot(xs, y)

# Title
ax2.set_title("samples from f(x) (by ARS)")

# Fix the plot size
ax2.set_xlim(-3*sigma, 3*sigma)
ax2.set_ylim(0,1100)

ax2.hist(samples, bins=75)

plt.show()

