#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, laplace, norm

# generalized student t with mu=0, q=1
# See Eqn 2 of "Bayesian sparsity path analysis"
def gt(x, a, c):
    return 1/(2*c) * (1+np.abs(x)/(a*c))**(-a-1)
    
x = np.linspace(-4, 4, 100)
n = norm.pdf(x, loc=0, scale=1)
l = laplace.pdf(x, loc=0, scale=1)
t = t.pdf(x, df=1, loc=0, scale=1)
g = gt(x, 1, 1)

plt.figure()
plt.plot(x, g, 'k:',
        x, l, 'b-',
        x, t, 'r--')
plt.legend(('GenStudent(a=1,b=1)', r'Laplace($\mu=0,\lambda=1)$',
    r'Student($\mu=0,\sigma=1,\nu=1$)'))
plt.ylabel('pdf')
plt.savefig('figures/genStudentLaplacePdfPlot.pdf')

plt.figure()
plt.plot(x, np.log(g), 'k:',
        x, np.log(l), 'b-',
        x, np.log(t), 'r--')
plt.legend(('GenStudent(a=1,b=1)', r'Laplace($\mu=0,\lambda=1)$',
    r'Student($\mu=0,\sigma=1,\nu=1$)'))
plt.ylabel('log pdf')
plt.savefig('figures/genStudentLaplaceLogPdfPlot.pdf')
  
plt.show()
