#!/usr/bin/env python

# Fit linear and quadratic surfaces to data
# Based on code by Romain Thibaux <thibaux@eecs.berkeley.edu>

import matplotlib.pyplot as pl
import numpy as np
import utils.util as util
from mpl_toolkits.mplot3d import Axes3D

data = util.load_mat('moteData/moteData.mat')
X = data['X']
y = data['y']

X_pad = util.add_ones(X)

for use_quad in (False, True):
  phi = X_pad

  if use_quad:
    phi = np.column_stack((X_pad, X**2))

  fig = pl.figure()
  ax = Axes3D(fig)
  ax.set_zlim(15, 19)
  ax.scatter(X[:,0], X[:,1], y)

  xrange = np.linspace(min(X[:,0]), max(X[:,0]), 10)
  yrange = np.linspace(min(X[:,1]), max(X[:,1]), 10)
  xx, yy = np.meshgrid(xrange, yrange)
  flatxx = xx.reshape((100, 1))
  flatyy = yy.reshape((100, 1))
  w = np.linalg.lstsq(phi, y)[0]

  z = util.add_ones(np.column_stack((flatxx, flatyy)))
  if use_quad:
    z = np.column_stack((z, flatxx**2, flatyy**2))

  z = np.dot(z , w)
  ax.plot_surface(xx, yy, z.reshape(10, 10),
                  rstride=1, cstride=1, cmap=pl.cm.hot)

  name = 'surfaceLinear.png'
  if use_quad:
    name = 'surfaceQuad.png'

  pl.savefig(name)
  pl.show()
