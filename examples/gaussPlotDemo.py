#!/usr/bin/env python3

# Plot the standard gaussian distribution.

import matplotlib.pyplot as pl
import numpy as np
from scipy.stats import norm

x = np.linspace(-3, 3, 100)
y = norm.pdf(x)
pl.plot(x, y)
pl.savefig('gaussPlotDemo.png')
pl.show()
