#!/usr/bin/env python

import matplotlib.pyplot as pl
import numpy as np
from numpy import tanh
from scipy.special import expit 

relu = lambda x: x * (x > 0)
funs = [tanh, expit, relu]
styles = ['r-', 'g:', 'k--']
labels = ['tanh', 'sigmoid', 'relu']

x = np.arange(-3, 3, .01)
for i, fun in enumerate(funs):
  pl.plot(x, fun(x), styles[i], label=labels[i])

pl.axis([-3, 3, -1, 3])
pl.legend()
pl.savefig('tanhPlot.png')
pl.show()
