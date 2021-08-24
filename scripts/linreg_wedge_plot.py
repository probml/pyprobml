
# Based on https://github.com/probml/pmtk3/blob/master/demos/linregWedgeDemo2.m

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml


from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

def gaussian(model, x, y, sigma):
  xx, yy = np.meshgrid(x, y)
  yypred = model.coef_ * xx + model.intercept_
  z = (yy - yypred) * (yy - yypred)
  z = np.exp(-0.5 * z)
  return z / ((2 * np.pi * sigma) ** 0.5)


np.random.seed(0)
N = 21
x = np.linspace(0.0, 20, N)
X0 = x.reshape(N,1)
X = np.c_[np.ones((N,1)), X0]
w = np.array([-1.5, 1/9.])
y =  w[0]*x + w[1]*np.square(x)
y = y + np.random.normal(0, 1, N) * 2

lin = LinearRegression(fit_intercept=True)
model = lin.fit(X0,y)
ypred = model.predict(X0)

fig = plt.figure()
ax = Axes3D(fig)
xrange = np.linspace(min(x), max(x), 30)
yrange = np.linspace(min(ypred), max(ypred), 30)
sigma = 1

z = gaussian(lin, xrange, yrange, sigma)
xx, yy = np.meshgrid(xrange, yrange)

surf = ax.plot_surface(
  xx, yy, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('P(Y|X)')
ax.view_init(azim=45)

pml.savefig('linregWedge.pdf')
plt.show()
