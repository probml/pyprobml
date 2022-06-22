# Quadratic lower bounds on the sigmoid (logistic) function
# Author : Aleyna Kara

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

sigmoid = lambda x : np.exp(x) / (1 + np.exp(x))
lambd = lambda x : 1/(2*x) * (sigmoid(x) - 1/2)

def sig_lower_JJ(xi, eta):
  lambd_xi = lambd(xi)
  c = -lambd_xi * xi**2 - 0.5 * xi + np.log(1 + np.exp(xi))
  bound = lambd_xi * eta**2 + 0.5 *(-eta) + c
  return np.exp(-bound)

def sig_lower_bohning(xi, eta):
  A, sig_xi = 0.25, sigmoid(xi)
  b = A * xi - sig_xi
  c = 0.5 * A * xi**2 - sig_xi * xi + np.log(1 + np.exp(xi))
  bound = 0.5 * A * eta**2 + b * eta + c
  return np.exp(-bound)

start, end, n = -6, 6, 121
x, xi = np.linspace(start, end, n), 2.5

plt.plot(x, sigmoid(x), 'r', linewidth=3)
plt.plot(x, sig_lower_JJ(xi, x), 'b:')
plt.plot([-xi, -xi], [0, sigmoid(-xi)], 'g', linewidth=3)
plt.plot([xi, xi], [0, sigmoid(xi)], 'g', linewidth=3)
plt.xlim([start, end])
plt.ylim(bottom=0)
plt.title(r'JJ bound $\chi$ = {}'.format(xi))
pml.savefig('JJBound.pdf', dpi=300)
plt.show()

xi = -2.5

plt.plot(x, sigmoid(x), 'r', linewidth=3)
plt.plot(x, sig_lower_bohning(xi, x),'b:')
plt.plot([2.6, 2.6], [0, sigmoid(-xi)], 'g', linewidth=3)
plt.xlim([start, end])
plt.ylim(bottom=0)
plt.title(r'Bohning bound $\chi$ = {}'.format(xi))
pml.savefig('bohningBound.pdf', dpi=300)
plt.show()