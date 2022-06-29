# Plot resisudal from 1d linear regression
# Based on https://github.com/probml/pmtk3/blob/master/demos/linregResiduals.m


import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
import pyprobml_utils as pml

np.random.seed(0)
N = 21
x = np.linspace(0.0, 20, N)
X0 = x.reshape(N, 1)
X = np.c_[np.ones((N, 1)), X0]
w = np.array([-1.5, 1/9.])
y = w[0]*x + w[1]*np.square(x)
y = y + np.random.normal(0, 1, N) * 2

w = np.linalg.lstsq(X, y,  rcond=None)[0]
# print(w)
y_estim = np.dot(X, w)

plt.plot(X[:, 1], y, 'o')
plt.plot(X[:, 1], y_estim, '-')
pml.savefig('linregResidualsNoBars.pdf')
plt.show()

for x0, y0, y_hat in zip(X[:, 1], y, y_estim):
    plt.plot([x0, x0], [y0, y_hat], 'k-')
plt.plot(X[:, 1], y, 'o')
plt.plot(X[:, 1], y_estim, '-')
plt.plot(X[:, 1], y_estim, 'x', color='r', markersize=12)
pml.savefig('linregResidualsBars.pdf')
plt.show()
