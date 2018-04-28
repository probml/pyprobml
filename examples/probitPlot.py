#!/usr/bin/env python

# Plots Sigmoid vs. Probit.

import matplotlib.pyplot as pl
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from utils.util import save_fig

x = np.arange(-6, 6, 0.1)
l = np.sqrt(np.pi/8); 
pl.plot(x, expit(x), 'r-', label='sigmoid')
pl.plot(x, norm.cdf(l*x), 'b--', label='probit')

pl.axis([-6, 6, 0, 1])
pl.legend()
save_fig('probitPlot.png')
pl.show()
