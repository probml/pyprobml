#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import t, laplace, norm

x = np.linspace(-4, 4, 100)
n = norm.pdf(x, loc=0, scale=1)
l = laplace.pdf(x, loc=0, scale=1 / (2 ** 0.5))
t = t.pdf(x, df=1, loc=0, scale=1)

pl.plot(n, 'k:',
        t, 'b--',
        l, 'r-')
pl.legend(('Gauss', 'Student', 'Laplace'))
pl.savefig('studentLaplacePdfPlot_1.png')

pl.figure()
pl.plot(np.log(n), 'k:',
        np.log(t), 'b--',
        np.log(l), 'r-')
pl.legend(('Gauss', 'Student', 'Laplace'))
pl.savefig('studentLaplacePdfPlot_2.png')

pl.show()
