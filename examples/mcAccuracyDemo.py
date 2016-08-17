#!/usr/bin/env python

import matplotlib.pyplot as pl
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import norm

x = np.arange(0.5, 2.5, 0.01)
for size in [10, 100, 1000]:
    samples = norm.rvs(loc=1.5, scale=0.5, size=size)
    y = norm.pdf(x, loc=1.5, scale=0.5)
    #draw pic
    pl.figure()
    pl.hist(samples, normed=True, rwidth=0.8)
    pl.plot(x, y, 'r')
    pl.xlim(0, 3)
    pl.title('n_samples = %d' % size)
    pl.savefig('mcAccuracyDemo_%d.png' % size)
    #draw kde pic
    kde = gaussian_kde(samples)
    y_estimate = kde(x)
    pl.figure()
    pl.plot(x, y, 'r', label='true pdf')
    pl.plot(x, y_estimate, 'b--', label='estimated pdf')
    pl.legend()
    pl.savefig('mcAccuracyDemo_kde%d.png' % size)
pl.show()
