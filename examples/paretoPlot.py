#!/usr/bin/env python

# Plots Pareto distribution

import matplotlib.pyplot as pl
import numpy as np
from scipy.stats import pareto

params = [(1, 3), (1, 2), (1, 1), (0.001, 1)]
styles = ['b-', 'r:', 'k-.', 'g--']
labels = ['m={:.2f}, k={:.2f}'.format(m, k) for m, k in params]

for i, param in enumerate(params):
  m, k = param
  probabilities = pareto.pdf(np.arange(0, 2, .01), k, scale=m)
  pl.plot(np.arange(0, 2, .01), probabilities, styles[i], label=labels[i])

pl.axis([0, 2, 0, 3])
pl.title('Pareto Distribution')
pl.legend()
pl.savefig('paretoPlot.png')
pl.show()
