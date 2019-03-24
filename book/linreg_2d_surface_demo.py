# Fit linear and quadratic surfaces to data
# Based on https://github.com/probml/pmtk3/blob/master/demos/surfaceFitDemo.m

import numpy as np
import matplotlib.pyplot as plt
import os

def save_fig(fname):
    figdir = os.path.join(os.environ["PYPROBML"], "figures")
    plt.tight_layout()    
    fullname = os.path.join(figdir, fname)
    print('saving to {}'.format(fullname))
    plt.savefig(fullname)
    
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

  xrange = np.linspace(min(X[:,0]), max(X[:,0]), 10)
  yrange = np.linspace(min(X[:,1]), max(X[:,1]), 10)
  xx, yy = np.meshgrid(xrange, yrange)
  flatxx = xx.reshape((100, 1))
  flatyy = yy.reshape((100, 1))
  w = np.linalg.lstsq(phi, y)[0]

  z = np.column_stack((flatxx, flatyy))
  z = np.column_stack((np.ones(100), z))
  if use_quad:
    z = np.column_stack((z, flatxx**2, flatyy**2))

  z = np.dot(z , w)
  ax.plot_surface(xx, yy, z.reshape(10, 10),
                  rstride=1, cstride=1, cmap='jet')

  name = 'linregSurfaceLinear.pdf'
  if use_quad:
    name = 'linregSurfaceQuad.pdf'

  save_fig(name)
  plt.show()
