# Gaussian Process Classifier demo
# Author: Drishtii@
# Based on
# https://github.com/probml/pmtk3/blob/master/demos/gpcDemo2d.m

# See also gpc_demo_2d_pytorch for a Gpytorch version of this demo.

import superimport

import pyprobml_utils as pml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# make synthetic data
np.random.seed(9)
n1=80
n2=40
S1 = np.eye(2)
S2 = np.array([[1, 0.95], [0.95, 1]])
m1 = np.array([0.75, 0]).reshape(-1, 1)
m2 = np.array([-0.75, 0])
xx = np.repeat(m1, n1).reshape(2, n1)
yy = np.repeat(m2, n2).reshape(2, n2)
x1 = np.linalg.cholesky(S1).T @ np.random.randn(2,n1) + xx
x2 = np.linalg.cholesky(S2).T @ np.random.randn(2,n2) + yy
x = np.concatenate([x1.T, x2.T])
y1 = -np.ones(n1).reshape(-1, 1)
y2 = np.ones(n2).reshape(-1, 1)
y = np.concatenate([y1, y2])
q = np.linspace(-4, 4, 81)
r = np.linspace(-4, 4, 81)
t1, t2 = np.meshgrid(q, r)
t = np.hstack([t1.reshape(-1, 1), t2.reshape(-1, 1)])

def g(x):
    return 5. - x[:, 1] - .5 * x[:, 0] ** 2
y_true = g(t)
y_true = y_true.reshape(81, 81)


def make_plot(gp):
  plt.figure()
  y_prob = gp.predict_proba(t)[:, 1]
  y_prob = y_prob.reshape(81, 81)
  plt.scatter(x1[0, :], x1[1, :], marker='o')
  plt.scatter(x2[0, :], x2[1, :], marker='+')
  plt.contour(t1, t2, y_prob, levels = np.linspace(0.1, 0.9, 9))
  plt.contour(t1, t2, y_prob, [0.5], colors=['red'])
  plt.title(gp.kernel_)

# GP without fitting the kernel hyper-parameters
# Note that 10.0 ~- 3.16**2
kernel = 10.0 * RBF(length_scale=0.5)
gp1 = GaussianProcessClassifier(kernel=kernel, optimizer=None)
gp1.fit(x, y)
make_plot(gp1)
pml.savefig('gpc2d_init_params.pdf')

# GP where we optimize the kernel parameters
gp2 = GaussianProcessClassifier(kernel=kernel)
gp2.fit(x, y)
make_plot(gp2)
pml.savefig('gpc2d_learned_params.pdf')