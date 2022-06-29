import superimport

from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import os
import pyprobml_utils as pml

x = np.linspace(0, 1, 100)
aa = [0.1, 1., 2., 8.]
bb = [0.1, 1., 3., 4.]
props = ['b-', 'r:', 'b-.', 'g--']
for a, b, p in zip(aa, bb, props):
    y = beta.pdf(x, a, b)
    plt.plot(y, p, lw=3, label='a=%.1f,b=%.1f' % (a, b))
plt.legend(loc='upper left')
pml.savefig('betaPlotDemo.png')
plt.show()
