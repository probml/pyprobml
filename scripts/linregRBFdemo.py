import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

np.random.seed(0)


def polyDataMake():
    '''
    sampling : thibaux
    '''
    xtrain = np.linspace(0, 20, 21)
    xtest = np.arange(0, 20, 0.1)
    sigma2 = 4
    w = np.array([-1.5, 1 / 9.0])
    ytrain = w[0] * xtrain + w[1] * xtrain ** 2 + np.random.rand(len(xtrain)) * np.sqrt(sigma2)
    ytestNoisefree = w[0] * xtest + w[1] * xtest ** 2
    ytestNoisy = ytestNoisefree + np.random.rand(len(xtest)) * np.sqrt(sigma2)
    return xtrain, ytrain, xtest, ytestNoisefree


def kernelRbfSigma(X1, X2, sigma):
    Z = 1.0 / np.sqrt(2 * np.pi * sigma ** 2)
    S = np.repeat(X1, len(X2), axis=1) - np.repeat(X2[np.newaxis, :], len(X1), axis=0)
    S = S ** 2

    K = Z * np.exp(-1 / (2 * sigma ** 2) * S)
    return K


[xtrain, ytrain, xtest, ytest] = polyDataMake()
l = 0.001
sigmas = np.array([0.5, 10, 50])
K = 10
centers = np.linspace(min(xtrain), max(xtrain), K)

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(len(sigmas)):

    clf = KernelRidge(alpha=l, kernel='rbf', gamma=sigmas[i])
    # clf = SVR(gamma = sigmas[i])
    clf.fit(xtrain[:, np.newaxis], ytrain[:, np.newaxis])
    ypred = clf.predict(ytest[:, np.newaxis])
    axs[i, 0].plot(xtrain, ytrain, '.b', markerSize=4)
    axs[i, 0].plot(xtest, ypred, 'k', linewidth=3)

    Xtest = kernelRbfSigma(xtest[:, np.newaxis], centers, sigmas[i])
    for j in range(K):
        axs[i, 1].plot(xtest, Xtest[:, j], 'b')

    XtrainRBF = kernelRbfSigma(xtrain[:, np.newaxis], centers, sigmas[i])
    axs[i, 2].imshow(XtrainRBF, cmap='gray')

fig.suptitle('linregRbfDemo')
plt.savefig('../figures/linregRbfDemo.png')