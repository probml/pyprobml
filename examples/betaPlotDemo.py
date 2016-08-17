#!/usr/bin/env python

from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as pl

x = np.linspace(0, 1, 100)
aa = [0.1, 1., 2., 8.]
bb = [0.1, 1., 3., 4.]
props = ['b-', 'r:', 'b-.', 'g--']
for a, b, p in zip(aa, bb, props):
    y = beta.pdf(x, a, b)
    pl.plot(y, p, lw=3, label='a=%.1f,b=%.1f' % (a, b))
pl.legend(loc='upper left')
pl.savefig('betaPlotDemo.png')
pl.show()
