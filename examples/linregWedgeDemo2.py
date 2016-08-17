#!/usr/bin/env python

# Plot Linear Gaussian CPD p(y|x) = N(Y|a + bx, sigma) 
# Here a is the offset and b is the slope. 

import matplotlib.pyplot as pl
import numpy as np
import sklearn
import utils.util as util
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

def gaussian(model, x, y, sigma):
  xx, yy = np.meshgrid(x, y)
  yypred = model.coef_ * xx + model.intercept_
 
  z = (yy - yypred) * (yy - yypred)
  z = np.exp(-0.5 * z)
  return z / ((2 * np.pi * sigma) ** 0.5)

N = 21
xtrain, ytrain, xtest, _, _, _ = util.poly_data_make(sampling='thibaux', n=N)

lin = LinearRegression(fit_intercept=True)
model = lin.fit(xtrain.reshape((N, 1)), ytrain)
ypred = model.predict(xtest.reshape((len(xtest), 1)))

fig = pl.figure()
ax = Axes3D(fig)
xrange = np.linspace(min(xtest), max(xtest), 300)
yrange = np.linspace(min(ypred), max(ypred), 300)
sigma = 1

z = gaussian(lin, xrange, yrange, sigma)
xx, yy = np.meshgrid(xrange, yrange)

surf = ax.plot_surface(
  xx, yy, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('P(Y|X)')
ax.view_init(azim=45)

pl.savefig('linregWedge2Wedge.png')
pl.show()
