#!/usr/bin/env python

import matplotlib.pyplot as pl
import numpy as np
from scipy.stats import poisson

for l in [1.0, 10.0]:
    pl.figure()
    probabilities = poisson.pmf(np.arange(30), l)
    pl.bar(np.arange(30), probabilities)
    pl.xticks(np.arange(0, 30, 5) + 0.4, np.arange(0, 30, 5))
    pl.title(r'$Poi (\lambda = %.2f)$' % l)
    pl.savefig('poissonPlotDemo_%s.png' % l)
pl.show()
