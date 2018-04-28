#!/usr/bin/env python

# Plot the gamma distribution

import matplotlib.pyplot as pl
import numpy as np
from scipy.stats import gamma
from utils.util import save_fig

x = np.linspace(0, 7, 100)
for a in [1, 1.5, 2]:
    y = gamma.pdf(x, a)
    pl.plot(x, y)
pl.legend(['a=%.1f' % a for a in [1, 1.5, 2]])
save_fig('gammaPlotDemo.png')
pl.show()
