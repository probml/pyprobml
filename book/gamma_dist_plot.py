# Plot the gamma distribution
# Based on https://github.com/probml/pmtk3/blob/master/demos/gammaPlotDemo.m

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from scipy.stats import gamma

x = np.linspace(0, 7, 100)
b = 1
plt.figure()
for a in [1, 1.5, 2]:
    y = gamma.pdf(x, a, scale=1/b, loc=0)
    plt.plot(x, y)
plt.legend(['a=%.1f, b=1' % a for a in [1, 1.5, 2]])
plt.title('Gamma(a,b) distributions')
save_fig('gammaDistb1.pdf')
plt.show()

x = np.linspace(0, 7, 100)
b = 1
a = 1
rv = gamma(a, scale=1/b, loc=0)
y = rv.pdf(x)
plt.plot(x, y)
plt.axvline(1, color='r')
plt.title('Gamma(1,1) distribution')
save_fig('gammaDist1.pdf')
plt.show()
