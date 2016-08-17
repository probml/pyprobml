#!/usr/bin/env python

# Plots the sigmoid function.

import matplotlib.pyplot as pl
import numpy as np

e = np.exp(1)
x = np.linspace(-10, 10, 1000)
y = e**x / (e**x + 1)
pl.plot(x, y)
pl.title('sigmoid function')
pl.savefig('sigmoidPlot.png')
pl.show()
