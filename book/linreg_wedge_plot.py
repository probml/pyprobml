
# Based on https://github.com/probml/pmtk3/blob/master/demos/linregWedgeDemo2.m

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


import sklearn
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

def gaussian(model, x, y, sigma):
  xx, yy = np.meshgrid(x, y)
  yypred = model.coef_ * xx + model.intercept_
 
  z = (yy - yypred) * (yy - yypred)
  z = np.exp(-0.5 * z)
  return z / ((2 * np.pi * sigma) ** 0.5)

def make_1dregression_data(n=21):
    np.random.seed(0)
    # Example from Romaine Thibaux
    xtrain = np.linspace(0.0, 20, n)
    xtest = np.arange(0.0, 20, 0.1)
    sigma2 = 4
    w = np.array([-1.5, 1/9.])
    fun = lambda x: w[0]*x + w[1]*np.square(x)
    # Apply function to make data
    ytrain = fun(xtrain) + np.random.normal(0, 1, xtrain.shape) * \
        np.sqrt(sigma2)
    ytestNoisefree = fun(xtest)
    ytestNoisy = ytestNoisefree + np.random.normal(0, 1, xtest.shape) * \
        np.sqrt(sigma2)
    return xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2

N = 21
xtrain, ytrain, xtest, _, _, _ = make_1dregression_data(N)

lin = LinearRegression(fit_intercept=True)
model = lin.fit(xtrain.reshape((N, 1)), ytrain)
ypred = model.predict(xtest.reshape((len(xtest), 1)))

fig = plt.figure()
ax = Axes3D(fig)
xrange = np.linspace(min(xtest), max(xtest), 30)
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

save_fig('linregWedge.pdf') 
plt.show()
