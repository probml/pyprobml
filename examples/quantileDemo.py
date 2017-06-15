#!/usr/bin/env python

import matplotlib.pyplot as pl
import numpy as np
from scipy.stats import norm

x = np.linspace(-3, 3, 100)
y = norm.pdf(x)
f = norm.cdf(x)

pl.figure()
pl.plot(x, f)
pl.title('CDF')
pl.savefig('quantileDemo_cdf.png')

pl.figure()
pl.plot(x, y)
pl.savefig('quantileDemo_gaussDemo.png')

x_sep_left = norm.ppf(0.025)
x_sep_right = norm.ppf(0.975)
x_fill_left = np.linspace(-3, x_sep_left, 100)
x_fill_right = np.linspace(x_sep_right, 3, 100)
pl.fill_between(x_fill_left,
                norm.pdf(x_fill_left),
                color='b')
pl.fill_between(x_fill_right,
                norm.pdf(x_fill_right),
                color='b')
pl.annotate(r'$\alpha/2$', xy=(x_sep_left, norm.pdf(x_sep_left)),
            xytext=(-2.5, 0.1),
            arrowprops=dict(facecolor='k'))
pl.annotate(r'$1-\alpha/2$', xy=(x_sep_right, norm.pdf(x_sep_right)),
            xytext=(2.5, 0.1),
            arrowprops=dict(facecolor='k'))
pl.ylim([0, 0.5])
pl.savefig('quantileDemo.png')
pl.show()
