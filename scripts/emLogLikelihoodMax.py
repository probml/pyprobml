import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os


if os.path.isdir('scripts'):
    os.chdir('scripts')


def f1(x): return np.log(multivariate_normal.pdf(x, 0, 0.25**2))+1
def f2(x): return np.log(multivariate_normal.pdf(x, 1, 0.2**2))+20


def f3(x): return 5*np.sin(2*(x-0.5)) + f1(0.5*x) + f2(0.5*x) + 3.5 + 20 * np.transpose(multivariate_normal.pdf(x, -2, 0.5**2)) - 20 * np.transpose(multivariate_normal.pdf(x, 3, 1**2)) - 70 * np.transpose(multivariate_normal.pdf(x,  4, 0.5**2)
                                                                                                                                                                                                             ) + 40 * np.transpose(multivariate_normal.pdf(x, -3, 0.5**2)) + 100 * np.transpose(multivariate_normal.pdf(x, -4, 0.8**2)) + 10 * np.transpose(multivariate_normal.pdf(x,  3, 0.3**2)) - 10 * np.transpose(multivariate_normal.pdf(x, -2.8, 0.5**2))


domain = np.arange(-5, 5.01, 0.01)

p1 = plt.plot(domain, f1(domain), '-b')
p2 = plt.plot(domain, f2(domain), ':g')
p3 = plt.plot(domain, f3(domain), '-.r')
plt.axis([-3, 5, -50, 50])
plt.legend(['Q('+'\u03B8'+','+'\u03B8'+'\u209C'+')', 'Q('+'\u03B8' +
            ','+'\u03B8'+'\u209C'+'+'+'\u2081'+')', 'I('+'\u03B8'+')'])
plt.vlines(-0.65, -50, -0.60, linestyles='dotted')
plt.vlines(0.065, -50, 8.766, linestyles='dotted')
plt.vlines(1.129, -50, 23.376, linestyles='dotted')

plt.xticks([], [])
plt.yticks([], [])

plt.text(-0.75, -58, '\u03B8'+'\u2096', fontsize=16)
plt.text(-0.165, -58, '\u03B8'+'\u209C'+'\u208A'+'\u2081', fontsize=16)
plt.text(1.029, -58, '\u03B8'+'\u209C'+'\u208A'+'\u2082', fontsize=16)



plt.savefig('../figures/emLogLikelihoodMax.png')
plt.show()
