# Plot the gamma distribution
# Based on https://github.com/probml/pmtk3/blob/master/demos/gammaPlotDemo.m

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")

x = np.linspace(0, 7, 100)
b = 1
plt.figure()
for a in [1, 1.5, 2]:
    y = gamma.pdf(x, a, scale=1/b, loc=0)
    plt.plot(x, y)
plt.legend(['a=%.1f, b=1' % a for a in [1, 1.5, 2]])
plt.title('Gamma(a,b) distributions')
plt.savefig(os.path.join(figdir, 'gammaDistb1.pdf'))
plt.show()

x = np.linspace(0, 7, 100)
b = 1
a = 1
rv = gamma(a, scale=1/b, loc=0)
y = rv.pdf(x)
plt.plot(x, y)
plt.axvline(1, color='r')
plt.title('Gamma(1,1) distribution')
plt.savefig(os.path.join(figdir, 'gammaDist1.pdf'))
plt.show()
