import superimport

import numpy as np
import matplotlib.pyplot as plt
import os

import pyprobml_utils as pml

xmin = -5
xmax = 0
ymin = -20
ymax = 150
domain = np.arange(xmin, xmax+0.01, 0.01)


def f(x): return - np.power(x, 3)


Xk = -4
def f1(x): return - 3 * np.power(x, 2)
def f2(x): return - 6 * x
def t(x): return f(Xk) + f1(Xk)*(x - Xk) + (1/2)*f2(Xk) * np.power((x - Xk), 2)


minNDX = np.argmin(t(domain))
minimum = domain[minNDX]

h1 = plt.plot(domain, f(domain), '-r')
h2 = plt.plot(domain, t(domain), '--b')
plt.plot(Xk, f(Xk), '.k')
plt.plot([Xk, Xk], [ymin, f(Xk)], ':k')
plt.plot(minimum, t(minimum), '.k')
plt.plot([minimum,  minimum], [ymin, t(minimum)], ':k')
plt.axis([xmin, xmax, ymin, ymax])
plt.xticks([], [])
plt.yticks([], [])
plt.text(-4.1, -30, '\u03B8'+'\u2096', fontsize=14)
plt.text(-2.1, -30, '\u03B8'+'\u2096'+'+'+'d'+'\u2096', fontsize=14)
pml.savefig(r'newtonsMethodMinQuadTheta.pdf')
plt.show()
