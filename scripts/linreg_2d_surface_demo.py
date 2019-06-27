# Fit linear and quadratic surfaces to data
# Based on https://github.com/probml/pmtk3/blob/master/demos/surfaceFitDemo.m

import numpy as np
import matplotlib.pyplot as plt
import os
from pyprobml_utils import save_fig

    
import scipy.io
from mpl_toolkits.mplot3d import Axes3D

datadir = os.path.join(os.environ["PYPROBML"], "data")
data = scipy.io.loadmat(os.path.join(datadir, 'moteData', 'moteData.mat'))
X = data['X']
y = data['y']

n = len(y)
X_pad = np.column_stack((np.ones(n), X))


for use_quad in (False, True):
  phi = X_pad

  if use_quad:
    phi = np.column_stack((X_pad, X**2))

  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_zlim(15, 19)
  ax.scatter(X[:,0], X[:,1], y, color='r')

  n = 10
  xrange = np.linspace(min(X[:,0]), max(X[:,0]), n)
  yrange = np.linspace(min(X[:,1]), max(X[:,1]), n)
  xx, yy = np.meshgrid(xrange, yrange)
  flatxx = xx.reshape((n**2, 1))
  flatyy = yy.reshape((n**2, 1))
  w = np.linalg.lstsq(phi, y)[0]

  z = np.column_stack((flatxx, flatyy))
  z = np.column_stack((np.ones(n**2), z))
  if use_quad:
    z = np.column_stack((z, flatxx**2, flatyy**2))

  f = np.dot(z , w)
  ax.plot_surface(xx, yy, f.reshape(n, n),
                  rstride=1, cstride=1, cmap='jet')

  name = 'linregSurfaceLinear.pdf'
  if use_quad:
    name = 'linregSurfaceQuad.pdf'

  save_fig(name)
  plt.show()
