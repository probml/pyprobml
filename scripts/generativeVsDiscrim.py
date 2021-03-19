import numpy as np
from numpy.random import laplace
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import os


if os.path.isdir('scripts'):
    os.chdir('./scripts')

domain = np.arange(0, 1.001, 0.001)


def f1(x): return multivariate_normal.pdf(x, 0.7, 0.09**2)


def f2(x): return 0.6*multivariate_normal.pdf(x, 0.1, 0.09**2) + \
    0.4*multivariate_normal.pdf(x, 0.4, 0.09**2)


plt.plot(domain, f1(domain), 'r:')
plt.plot(domain, f2(domain), '-k')
plt.xlabel('x')
plt.ylabel('class conditional densities')
plt.annotate('p(x|y=1)', (0.196, 2))
plt.annotate('p(x|y=2)', (0.8, 4))
plt.savefig('../figures/genVsDiscrimClassCond')

plt.show()

domain = np.arange(0, 1.001, 0.001)

def f1(x): return 1/(1 + np.exp((27*x-15)))
def f2(x): return 1/(1 + np.exp((-27*x+15)))


plt.plot(domain, f1(domain), ':r')
plt.plot(domain, f2(domain), '-k')
plt.plot([0.556,0.556],[0,1.2],'-g')
plt.annotate('p(y=1|x)', (0.24, 1.1))
plt.annotate('p(y=2|x)', (0.9, 1.1))
plt.savefig('../figures/genVsDiscrimPost')

plt.show()
