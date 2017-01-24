#!/usr/bin/env python

# Plots 0-1, hinge and log loss.

import matplotlib.pyplot as pl
import numpy as np

zeroOne = np.vectorize(lambda x: 1 * (x <= 0))
hinge = np.vectorize(lambda x: max(0, 1-x))
logLoss =  np.vectorize(lambda x: np.log2(1 + np.exp(-x)))

funs = [zeroOne, hinge, logLoss]
styles = ['k-', 'b:', 'r-.']
labels = ['0-1', 'hinge', 'logloss']
x = np.arange(-2, 2, .01)

for i, fun in enumerate(funs):
  pl.plot(x, fun(x), styles[i], label=labels[i])

pl.axis([-2.1, 2.1, -0.1, 3.1])
pl.legend()
pl.savefig('hingeLossPlot.png')
pl.show()
