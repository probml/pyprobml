#!/usr/bin/env python3

# Linear regression on data with residuals.

import matplotlib.pyplot as pl
import numpy as np
import utils.util as util

N = 21
x, y, _, _, _, _ = util.poly_data_make(sampling='thibaux', n=N)
X = np.concatenate((np.ones((N,1)), x.reshape(N,1)), axis=1)  
w = np.linalg.lstsq(X, y)[0]
y_estim = np.dot(X,w)

pl.plot(X[:,1], y, 'o')
pl.plot(X[:,1], y_estim, '-')
pl.savefig('linregResidualsNoBars.png')
pl.show()

for x0, y0, y_hat in zip(X[:,1], y, y_estim):
  pl.plot([x0, x0],[y0, y_hat],'k-')
pl.plot(X[:,1], y, 'o')
pl.plot(X[:,1], y_estim, '-')

pl.savefig('linregResidualsBars.png')
pl.show()
