#!/usr/bin/env python

import matplotlib.pyplot as pl
import numpy as np
from scipy.stats import t, laplace, norm

a = np.random.randn(30)
outliers = np.array([8, 8.75, 9.5])
pl.hist(a, 7, weights=[1 / 30] * 30, rwidth=0.8)

#fit without outliers
x = np.linspace(-5, 10, 500)

loc, scale = norm.fit(a)
n = norm.pdf(x, loc=loc, scale=scale)

loc, scale = laplace.fit(a)
l = laplace.pdf(x, loc=loc, scale=scale)

fd, loc, scale = t.fit(a)
s = t.pdf(x, fd, loc=loc, scale=scale)
pl.plot(x, n, 'k>',
        x, s, 'r-',
        x, l, 'b--')
pl.legend(('Gauss', 'Student', 'Laplace'))
pl.savefig('robustDemo_without_outliers.png')

#add the outliers
pl.figure()
pl.hist(a, 7, weights=[1 / 33] * 30, rwidth=0.8)
pl.hist(outliers, 3, weights=[1 / 33] * 3, rwidth=0.8)
aa = np.hstack((a, outliers))

loc, scale = norm.fit(aa)
n = norm.pdf(x, loc=loc, scale=scale)

loc, scale = laplace.fit(aa)
l = laplace.pdf(x, loc=loc, scale=scale)

fd, loc, scale = t.fit(aa)
t = t.pdf(x, fd, loc=loc, scale=scale)
pl.plot(x, n, 'k:',
        x, t, 'r-',
        x, l, 'b--')
pl.legend(('Gauss', 'Student', 'Laplace'))
pl.savefig('robustDemo_with_outliers.png')
pl.show()
