# Plot error surface for linear regression model.
# Based on https://github.com/probml/pmtk3/blob/master/demos/contoursSSEdemo.m


import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from mpl_toolkits.mplot3d import Axes3D

def make_1dregression_data(n=21):
    np.random.seed(0)
    # Example from Romaine Thibaux
    xtrain = np.linspace(0.0, 20, n)
    xtest = np.arange(0.0, 20, 0.1)
    sigma2 = 1
    w = np.array([-1.5, 1/9.])
    fun = lambda x: w[0]*x + w[1]*np.square(x)
    # Apply function to make data
    ytrain = fun(xtrain) + np.random.normal(0, 1, xtrain.shape) * \
        np.sqrt(sigma2)
    ytestNoisefree = fun(xtest)
    ytestNoisy = ytestNoisefree + np.random.normal(0, 1, xtest.shape) * \
        np.sqrt(sigma2)
    return xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2

def add_ones(X):
    """Add a column of ones to X"""
    n = len(X)
    return np.column_stack((np.ones(n), X))

X,y,_,_,_,_ = make_1dregression_data()
X = add_ones(X)

N = len(y)
w = np.linalg.lstsq(X, y)[0]
v = np.arange(-6, 6, .1)
W0, W1 = np.meshgrid(v, v)

SS = np.array([sum((w0*X[:,0] + w1*X[:,1] - y)**2) for w0, w1 in zip(np.ravel(W0), np.ravel(W1))])
SS = SS.reshape(W0.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W0, W1, SS)
save_fig('linregSurfSSE.pdf')
plt.show()

fig,ax = plt.subplots()
ax.set_title('Sum of squares error contours for linear regression')
CS = pl.contour(W0, W1, SS)
pl.plot([-4.351],[0.5377],'x')
save_fig('linregContoursSSE.pdf')
plt.show()