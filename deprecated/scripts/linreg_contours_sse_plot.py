# Plot error surface for linear regression model.
# Based on https://github.com/probml/pmtk3/blob/master/demos/contoursSSEdemo.m

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

from mpl_toolkits.mplot3d import axes3d, Axes3D 

np.random.seed(0)

N = 21
x = np.linspace(0.0, 20, N)
X0 = x.reshape(N,1)
X = np.c_[np.ones((N,1)), X0]
w = np.array([-1.5, 1/9.])
y =  w[0]*x + w[1]*np.square(x)
y = y + np.random.normal(0, 1, N) * 2

w = np.linalg.lstsq(X, y, rcond=None)[0]
W0, W1 = np.meshgrid(np.linspace(-8,0,100), np.linspace(-0.5,1.5,100))

SS = np.array([sum((w0*X[:,0] + w1*X[:,1] - y)**2) for w0, w1 in zip(np.ravel(W0), np.ravel(W1))])
SS = SS.reshape(W0.shape)

plt.figure()
plt.contourf(W0, W1, SS)
pml.savefig('linregHeatmapSSE.pdf')
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W0, W1, SS)
pml.savefig('linregSurfSSE.pdf')
plt.show()

fig,ax = plt.subplots()
CS = plt.contour(W0, W1, SS, levels=np.linspace(0,2000,10), cmap='jet')
plt.plot(w[0], w[1],'x')
pml.savefig('linregContoursSSE.pdf')
plt.show()