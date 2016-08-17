#!/usr/bin/env python3

# Error surface for linear regression model.

import matplotlib.pyplot as pl
import numpy as np
import utils.util as util
from mpl_toolkits.mplot3d import Axes3D

def contoursSSEDemo():
  N = 21
  x,y,_,_,_,_ = util.poly_data_make(sampling='thibaux', n=N)
  X = util.add_ones(x)

  return X,y

if __name__ == '__main__':
  X,y  = contoursSSEDemo()
  N = len(y)
  w = np.linalg.lstsq(X, y)[0]
  v = np.arange(-6, 6, .1)
  W0, W1 = np.meshgrid(v, v)
  
  SS = np.array([sum((w0*X[:,0] + w1*X[:,1] - y)**2) for w0, w1 in zip(np.ravel(W0), np.ravel(W1))])
  SS = SS.reshape(W0.shape)
  
  fig = pl.figure()
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(W0, W1, SS)
  pl.savefig('linregSurfSSE.png')
  pl.show()
  
  fig,ax = pl.subplots()
  ax.set_title('Sum of squares error contours for linear regression')
  CS = pl.contour(W0, W1, SS)
  pl.plot([-4.351],[0.5377],'x')  

  pl.savefig('linregContoursSSE.png')
  pl.show()
