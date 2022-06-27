import superimport

import numpy as np
from scipy.spatial.distance import cdist
import math
import matplotlib.pyplot as plt
from cycler import cycler
import pyprobml_utils as pml

CB_color = ['#377eb8', '#ff7f00', '#4daf4a']

cb_cycler = (cycler(linestyle=['-', '--', '-.']) * cycler(color=CB_color))
plt.rc('axes', prop_cycle=cb_cycler)

np.random.seed(0)
N = 100
x = 10 * (np.linspace(-1, 1, 100).reshape(-1, 1))
ytrue = np.array([math.sin(abs(el)) / (abs(el)) for el in x]).reshape(-1, 1)
noise = 0.1
y = ytrue + noise * np.random.randn(N, 1)
x = (x - x.mean()) / x.std()  # normalizing.


plt.plot(x, ytrue)
plt.plot(x, y, 'kx')


def rbf_features(X, centers, sigma):
    dist_mat = cdist(X, centers, 'minkowski', p=2.)
    return np.exp((-0.5 / (sigma ** 2)) * (dist_mat ** 2))


# Nadaraya-Watson Kernel Regressor
# using rbf kernel with autosSelected bandwidth given a range.
class NdwkernelReg:

    def __init__(self, gammas=None):
        self.gammas = gammas
        self.gamma = None

    def fit(self, X, y):
        self.X = X
        self.y = y

        self.gamma = self.select_gamma(self.gammas)

    def predict(self, X):
        K = rbf_features(self.X, X, self.gamma)
        return (K * self.y).sum(axis=0) / K.sum(axis=0)

    # leave-one-out crossValidation
    def select_gamma(self, gammas):
        mse = []

        for gamma in gammas:
            K = rbf_features(self.X, self.X, gamma)
            K = K - np.diag(np.diag(K))  # vanishing the diagonal elements
            y_pred = (K * self.y).sum(axis=0) / K.sum(axis=0)
            mse.append(((y_pred[:, np.newaxis] - self.y) ** 2).mean())

        return gammas[np.argmin(mse)]


nws = NdwkernelReg(gammas=np.linspace(0.1, 1, 10))
nws.fit(x, y)
y_estimate = nws.predict(x)
plt.plot(x, y_estimate)
plt.legend(['true', 'data', 'estimate'], fontsize=12)
pml.savefig("kernelRegressionDemo.pdf")
plt.show()
