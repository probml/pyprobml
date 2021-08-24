import superimport

import numpy as np
from numpy.linalg import cholesky
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import pyprobml_utils as pml

np.random.seed(10)


def gaussSample(mu, sigma, n):
    A = cholesky(sigma)
    Z = np.random.normal(loc=0, scale=1, size=(len(mu), n))
    return np.dot(A, Z).T + mu


mtrue = {}
prior = {}
muTrue = np.array([0.5, 0.5])
Ctrue = 0.1 * np.array([[2, 1], [1, 1]])
mtrue['mu'] = muTrue
mtrue['Sigma'] = Ctrue
xyrange = np.array([[-1, 1], [-1, 1]])
ns = [10]
X = gaussSample(mtrue['mu'], mtrue['Sigma'], ns[-1])


#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
#fig.suptitle('gauss2dUpdateData')
fig, ax1 = plt.subplots()
ax1.plot(X[:, 0], X[:, 1], 'o', markersize=8, markerfacecolor='b')
ax1.set_ylim([-1, 1])
ax1.set_xlim([-1, 1])
ax1.set_title('data')
ax1.plot(muTrue[0], muTrue[1], 'x', linewidth=5, markersize=20, color='k')
pml.savefig("gauss_2d_update_data.pdf")


prior['mu'] = np.array([0, 0])
prior['Sigma'] = 0.1 * np.eye(2)

npoints = 100j
out = np.mgrid[xyrange[0, 0]:xyrange[0, 1]:npoints, xyrange[1, 0]:xyrange[1, 1]:npoints]
X1, X2 = out[0], out[1]
nr = X1.shape[0]
nc = X2.shape[0]
points = np.vstack([np.ravel(X1), np.ravel(X2)]).T
p = multivariate_normal.pdf(points, mean=prior['mu'], cov=prior['Sigma']).reshape(nr, nc)
fig, ax2 = plt.subplots()
ax2.contour(X1, X2, p)
ax2.set_ylim([-1, 1])
ax2.set_xlim([-1, 1])
ax2.set_title('prior')
pml.savefig("gauss_2d_update_prior.pdf")


post = {}
data = X[:ns[0], :]
n = ns[0]
S0 = prior['Sigma']
S0inv = inv(S0)
S = Ctrue
Sinv = inv(S)
Sn = inv(S0inv + n * Sinv)
mu0 = prior['mu']
xbar = np.mean(data, 0)
muN = np.dot(Sn, (np.dot(n, np.dot(Sinv, xbar)) + np.dot(S0inv, mu0)))

post['mu'] = muN
post['Sigma'] = Sn


p = multivariate_normal.pdf(points, mean=post['mu'], cov=post['Sigma']).reshape(nr, nc)
fig, ax3 = plt.subplots()
ax3.contour(X1, X2, p)
ax3.set_ylim([-1, 1])
ax3.set_xlim([-1, 1])
ax3.set_title('post after 10 observation')
pml.savefig("gauss_2d_update_post.pdf")

#fig.savefig("gauss2dUpdatePostSubplot.pdf")