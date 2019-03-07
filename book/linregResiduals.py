#!/usr/bin/env python3
# Linear regression on data with residuals.
# Based on https://github.com/probml/pmtk3/blob/master/demos/linregResiduals.m


import matplotlib.pyplot as plt
import numpy as np
#from utils import util

N = 21
#x, y, _, _, _, _ = util.poly_data_make(sampling='thibaux', n=N)

x = np.linspace(1, 20, N)
sigma2 = 1
w = np.array([1, 1])
fun = lambda x: w[0] + w[1]*x 
y = fun(x) + np.random.normal(0, 1, x.shape) * np.sqrt(sigma2)   

X = np.concatenate((np.ones((N,1)), x.reshape(N,1)), axis=1)  
w = np.linalg.lstsq(X, y)[0]
y_estim = np.dot(X,w)

plt.plot(X[:,1], y, 'o')
plt.plot(X[:,1], y_estim, '-')
plt.savefig('linregResidualsNoBars.png')
plt.show()

for x0, y0, y_hat in zip(X[:,1], y, y_estim):
  plt.plot([x0, x0],[y0, y_hat],'k-')
plt.plot(X[:,1], y, 'o')
plt.plot(X[:,1], y_estim, '-')

plt.savefig('linregResidualsBars.png')
plt.show()
