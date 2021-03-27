import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

np.random.seed(654321)


def fun(x, w):
    return w[0] * x + w[1] * np.square(x)


# 'Data as mentioned in the matlab code'
def polydateMake():
    n = 21
    sigma = 2
    xtrain = np.linspace(0, 20, n)
    xtest = np.arange(0, 20.1, 0.1)
    w = np.array([-1.5, 1 / 9])
    ytrain = fun(xtrain, w).reshape(-1, 1) + np.random.randn(xtrain.shape[0], 1)
    ytestNoisefree = fun(xtest, w)
    ytestNoisy = ytestNoisefree + sigma * np.random.randn(xtest.shape[0], 1) * sigma

    return xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy


[xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy] = polydateMake()

sigmas = [0.5, 10, 50]
K = 10
centers = np.linspace(np.min(xtrain), np.max(xtrain), K)


def addones(x):
    # x is of shape (s,)
    return np.insert(x[:, np.newaxis], 0, [[1]], axis=1)


def pairwise_kernel(X, Y, sigma):
    Z = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    pairwise_diff = np.repeat(X[:, np.newaxis], Y.shape[0], axis=1) - np.repeat(Y[np.newaxis, :], X.shape[0], axis=0)
    return Z * (np.exp((-0.5 / (sigma ** 2)) * (pairwise_diff ** 2)))


fig, ax = plt.subplots(3, 3, figsize=(6, 6))
plt.tight_layout()

xtest = np.delete(xtest, 10 * xtrain) # xtrain data points are deleted, to avoid overfitted predictions.

for (i, s) in enumerate(sigmas):
    kernel = RBF(s)
    reg = GaussianProcessRegressor(kernel=kernel, random_state=2, alpha=0.088)  # alpha for numerical stability
    reg.fit(addones(xtrain), ytrain)
    
    ypred = reg.predict(addones(xtest))
    ax[i, 0].plot(xtrain, ytrain, 'b.', markersize=8)
    ax[i, 0].plot(xtest, ypred, 'k')
    ax[i, 0].set_ylim([-10, 20])
    ax[i, 0].set_xticks(np.arange(0, 21, 5))

    Ktest = pairwise_kernel(xtest, centers, s)
    for j in range(K):
        ax[i, 1].plot(xtest, Ktest[:, j], 'b')
        ax[i, 1].set_xticks(np.arange(0, 21, 5))
        ax[i, 1].ticklabel_format(style='sci', scilimits=(-2, 2))

    Ktrain = pairwise_kernel(xtrain, centers, s)
    ax[i, 2].imshow(Ktrain, cmap='gray')
    ax[i, 2].set_xticks(np.arange(0, 11, 2))
    plt.show()
plt.savefig("../figures/rbfDemoALL.pdf", dpi=300)
