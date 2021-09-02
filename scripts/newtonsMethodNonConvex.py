import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

xmin = -4
xmax = 13
ymin = -50
ymax = 600
domain = np.arange(xmin, xmax+0.01, 0.01)
def f(x): return -np.power(x, 3) + 15 * np.power(x, 2)


Xk = 8
def f1(x): return - 3 * np.power(x, 2) + 30*x
def f2(x): return - 6*x + 30


def t(x): return f(Xk) + f1(Xk) * (x - Xk) + (1/2)*f2(Xk)*np.power((x - Xk), 2)


val = np.max(t(domain))
maxNDX = np.argmax(t(domain))
maximum = domain[maxNDX]

h1 = plt.plot(domain, f(domain), '-r')
h2 = plt.plot(domain, t(domain), '--b')
plt.plot(Xk, f(Xk), '.k')
plt.plot([Xk, Xk], [ymin, f(Xk)], ':k')
plt.plot(maximum, t(maximum), '.k')
plt.plot([maximum,  maximum], [ymin, t(maximum)], ':k')
plt.axis([xmin, xmax, ymin, ymax])
plt.xticks([], [])
plt.yticks([], [])
plt.text(7.8, -90, '\u03B8'+'\u2096', fontsize=14)
plt.text(10.3, -90, '\u03B8'+'\u2096'+'+'+'d'+'\u2096', fontsize=14)
pml.savefig(r'newtonsMethodNonConvexTheta.pdf')
plt.show()
