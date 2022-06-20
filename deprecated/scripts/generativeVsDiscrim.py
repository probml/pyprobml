import superimport

import numpy as np
from numpy.random import laplace
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pyprobml_utils as pml

domain = np.arange(0, 1.001, 0.001)

def f1(x): return multivariate_normal.pdf(x, 0.7, 0.09**2)


def f2(x): return 0.6*multivariate_normal.pdf(x, 0.1, 0.09**2) + \
    0.4*multivariate_normal.pdf(x, 0.4, 0.09**2)


plt.plot(domain, f1(domain), 'r-', linewidth=2)
plt.plot(domain, f2(domain), 'k-', linewidth=2)
plt.xlabel('x')
plt.ylabel('class conditional densities')
plt.annotate('p(x|y=1)', (0.196, 2), fontsize=14)
plt.annotate('p(x|y=2)', (0.8, 4), fontsize=14)
pml.savefig('genVsDiscrimClassCond.pdf')
plt.show()

domain = np.arange(0, 1.001, 0.001)


def f1(x): return 1/(1 + np.exp((27*x-15)))
def f2(x): return 1/(1 + np.exp((-27*x+15)))


plt.plot(domain, f1(domain), '-r', linewidth=2)
plt.plot(domain, f2(domain), '-k', linewidth=2)
plt.plot([0.556, 0.556], [0, 1.2], '-g')
plt.annotate('p(y=1|x)', (0.14, 1.1), fontsize=14)
plt.annotate('p(y=2|x)', (0.8, 1.1), fontsize=14)
pml.savefig('genVsDiscrimPost.pdf')
plt.show()


domain = np.arange(0.001, 10.001, 0.001)
def NB(x): return np.power(x, -0.5)
def LR(x): return np.divide(7, np.power(x, 0.8))-2


plt.axis([-0.1, 10, -1, 10])

plt.plot(domain, NB(domain), '--r')
plt.plot(domain, LR(domain), '-b')
plt.xlabel('size of training set')
plt.ylabel('test error')
plt.xticks([])
plt.yticks([])
plt.annotate('NB', (0.5, 1.5), fontsize=20, color='red')
plt.annotate('LR', (1.8, 3.5), fontsize=20, color='blue')
pml.savefig('genVsDiscrimTestError.pdf')
plt.show()
