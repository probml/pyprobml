# Plot the gamma distribution
# Based on https://github.com/probml/pmtk3/blob/master/demos/gammaPlotDemo.m

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

from scipy.stats import gamma


x = np.linspace(0, 7, 100)
aa = [1.0, 1.5, 2.0, 1.0, 1.5, 2.0]
bb = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
#props = ['b-', 'r:', 'k-.', 'g--', 'c-', 'o-']
props = ['b-', 'r-', 'k-', 'b:', 'r:', 'k:']

for a, b, p in zip(aa, bb, props):
    y = gamma.pdf(x, a, scale=1/b, loc=0)
    plt.plot(x, y, p, lw=3, label='a=%.1f,b=%.1f' % (a, b))
plt.title('Gamma distributions')
plt.legend(fontsize=14)
pml.savefig('gammadist.pdf')
plt.show()


x = np.linspace(0, 7, 100)
b = 1
plt.figure()
for a in [1, 1.5, 2]:
    y = gamma.pdf(x, a, scale=1/b, loc=0)
    plt.plot(x, y)
plt.legend(['a=%.1f, b=1' % a for a in [1, 1.5, 2]])
plt.title('Gamma(a,b) distributions')
pml.savefig('gammaDistb1.pdf')
plt.show()

x = np.linspace(0, 7, 100)
b = 1
a = 1
rv = gamma(a, scale=1/b, loc=0)
y = rv.pdf(x)
plt.plot(x, y)
plt.axvline(1, color='r')
plt.title('Gamma(1,1) distribution')
pml.savefig('gammaDist1.pdf')
plt.show()
