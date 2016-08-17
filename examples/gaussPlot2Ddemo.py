#!/usr/bin/env python

# Plots 2D gaussian contours.

import matplotlib.pyplot as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def mvn2d(x, y, sigma):
    xx, yy = np.meshgrid(x, y)
    u = np.array([np.mean(x), np.mean(y)])
    xy = np.c_[xx.ravel(), yy.ravel()]
    sigma_inv = np.linalg.inv(sigma)
    z = np.dot((xy - u), sigma_inv)
    z = np.sum(z * (xy - u), axis=1)
    z = np.exp(-0.5 * z)
    return z / (2 * np.pi * np.linalg.det(sigma) ** 0.5)

fig = pl.figure()
ax = Axes3D(fig)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
sigma = np.array([[1, 0], [0, 1]])
z = mvn2d(x, y, sigma)
xx, yy = np.meshgrid(x, y)

#plot figure
ax.plot_surface(xx, yy, z.reshape(100, 100),
                rstride=1, cstride=1, cmap=pl.cm.hot)
pl.savefig('gaussPlot2Ddemo_1.png')
pl.figure()
pl.contour(xx, yy, z.reshape(100, 100))
pl.savefig('gaussPlot2Ddemo_2.png')

sigma1 = np.array([[2, 0], [0, 1]])
z = mvn2d(x, y, sigma1)
pl.figure()
pl.contour(xx, yy, z.reshape(100, 100))
pl.savefig('gaussPlot2Ddemo_3.png')

sigma2 = np.array([[1, 1], [0, 1]])
z = mvn2d(x, y, sigma2)
pl.figure()
pl.contour(xx, yy, z.reshape(100, 100))
pl.savefig('gaussPlot2Ddemo_4.png')
pl.show()
