import superimport

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cholesky
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(12)


def gaussSample(mu, sigma, n):
    A = cholesky(sigma)
    Z = np.random.normal(loc=0, scale=1, size=(len(mu), n))
    return np.dot(A, Z).T + mu


def sqDistance(p, q):
    pSOS = np.sum(p ** 2, axis=1)
    qSOS = np.sum(q ** 2, axis=1)
    pSOS = np.repeat(pSOS[..., np.newaxis], len(qSOS), axis=1)
    dist = pSOS + qSOS - 2 * np.dot(p, q.T)
    return dist


def kernelRbfSigma(X1, X2, sigma):
    Z = 1.0 / np.sqrt(2 * np.pi * sigma ** 2)
    S = sqDistance(X1, X2)
    K = Z * np.exp(-1 / (2 * sigma ** 2) * S)
    return K


def createXORdata(doplot=False):
    off1 = gaussSample([1, 1], 0.5 * np.eye(2), 20)
    off2 = gaussSample([5, 5], 0.5 * np.eye(2), 20)
    on1 = gaussSample([1, 5], 0.5 * np.eye(2), 20)
    on2 = gaussSample([5, 1], 0.5 * np.eye(2), 20)
    X = np.concatenate([off1, off2, on1, on2], axis=0)
    y = np.concatenate([np.zeros((len(off1) + len(off2))), np.ones((len(on1) + len(on2)))], axis=0)

    if doplot:
        plt.plot(X[y == 0, 0], X[y == 0, 1], 'ob', MarkerSize=8)
        plt.plot(X[y == 1, 0], X[y == 1, 1], '+r', MarkerSize=8)
        plt.show()
    return X, y


def addOnes(X):
    X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
    return X


def degexpand(x, deg, addO=0):
    n, m = x.shape

    temp = x
    for i in range(1, deg):
        xx = np.concatenate((x, temp ** (i + 1)), axis=1)
        x = xx

    if addO:
        xx = addOnes(xx)

    return xx


def rescaleData(x):
    minVal = -1
    maxVal = 1

    minx = np.min(X, axis=0)
    rangex = np.max(X, axis=0) - np.min(X, axis=0)
    y = (x - minx) / rangex

    y = y * (maxVal - minVal)
    y = y + minVal
    return y


def rbf_prototype(X1, protoTypesStnd, rbfScale):
    X1 = (X1 - X1.mean(axis=0)) / X1.std(axis=0)
    X1 = kernelRbfSigma(X1, protoTypesStnd, rbfScale)
    X1 = addOnes(X1)
    return X1


def poly_data(X1, deg):
    X1 = rescaleData(X1)
    X1 = degexpand(X1, deg)
    X1 = addOnes(X1)
    return X1


X, y = createXORdata(False)
tol = 1e-2

model = LogisticRegression(tol=tol)
model.fit(X, y)
ypred = model.predict(X)
errorRate = np.mean(ypred != y)

clf = LogisticRegression()
clf.fit(X, y)

fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
# Retrieve the model parameters.
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b / w2
m = -w1 / w2

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xd = np.array([x_min, x_max])
yd = m * xd + c
ax1.plot(xd, yd, 'k', lw=1, ls='--')
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot_raw = np.c_[xx.ravel(), yy.ravel()]

Z = clf.predict(X_plot_raw)

# Put the result into a color plot
Z = Z.reshape(xx.shape)

ax1.pcolormesh(xx, yy, Z, cmap='Set3')

# Plot also the training points
ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())
ax1.set_ylabel(r'$x_2$')
ax1.set_xlabel(r'$x_1$')
ax1.set_title('Simple Logistic regression')
fig.savefig('../figures/logregXorLinear.png')

rbfScale = 1
polydeg = 2
protoTypes = np.array([[1,1], [1,5], [5,1], [5,5]])

protoTypesStnd = (protoTypes - protoTypes.mean(axis=0)) / protoTypes.std(axis=0)

X_rbf = rbf_prototype(X, protoTypesStnd, rbfScale)
lr = LogisticRegression(tol=tol)
lr.fit(X_rbf, y)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot_raw = np.c_[xx.ravel(), yy.ravel()]
X_plot = rbf_prototype(X_plot_raw, protoTypesStnd, rbfScale)
Z = lr.predict(X_plot)

# Put the result into a color plot
Z = Z.reshape(xx.shape)

fig, ax2 = plt.subplots(1, 1, figsize=(9, 6))
ax2.pcolormesh(xx, yy, Z, cmap='Set3')

# Plot also the training points
ax2.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
ax2.scatter(protoTypes[:, 0], protoTypes[:, 1], marker='*', c='r', s=90)
ax2.set_xlabel(r'$x_1$')
ax2.set_ylabel(r'$x_2$')

ax2.set_xlim(xx.min(), xx.max())
ax2.set_ylim(yy.min(), yy.max())
ax2.set_title('rbf prototypes')
fig.savefig('../figures/logregXorRbfProto.png')

X_poly = poly_data(X, 10)
lr = LogisticRegression(tol=tol)
lr.fit(X_poly, y)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot_raw = np.c_[xx.ravel(), yy.ravel()]
X_plot = poly_data(X_plot_raw, 10)
Z = lr.predict(X_plot)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig, ax3 = plt.subplots(1, 1, figsize=(9, 6))
ax3.pcolormesh(xx, yy, Z, cmap='Set3')

# Plot also the training points
ax3.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
ax3.set_xlabel(r'$x_1$')
ax3.set_ylabel(r'$x_2$')
ax3.set_title('poly10')
ax3.set_xlim(xx.min(), xx.max())
ax3.set_ylim(yy.min(), yy.max())
fig.savefig('../figures/logregXorPoly.png')