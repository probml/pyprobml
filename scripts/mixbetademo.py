import numpy as np
from scipy.special import betaln, betainc
from scipy.special import logsumexp
from matplotlib import pyplot as plt


def normalizeLogspace(x):
    L = logsumexp(x, 0)
    y = x - L
    return y


def evalpdf(thetas, postZ, alphaPost):
    p = np.zeros_like(thetas)
    # print(p.shape)
    M = np.size(postZ)
    for k in range(M):
        a = alphaPost[k, 0]
        b = alphaPost[k, 1]
        p += postZ[k] * np.exp(betaLogprob(a, b, thetas))

    return p


def betaLogprob(a, b, X):
    logkerna = (a - 1) * np.log(X)
    logkerna[a == 1 and X == 0] = 0
    logkernb = (b - 1) * np.log(1 - X)
    logkernb[b == 1 and X == 1] = 0
    logp = logkerna + logkernb - betaln(a, b)

    return logp


dataSS = np.array([20, 10])
alphaPrior = np.array([[20, 20], [30, 10]])
M = 2
mixprior = np.array([0.5, 0.5])
logmarglik = np.zeros((2,))
for i in range(M):
    logmarglik[i] = betaln(alphaPrior[i, 0] + dataSS[0], alphaPrior[i, 1] + dataSS[1]) \
                    - betaln(alphaPrior[i, 0], alphaPrior[i, 1])

mixpost = np.exp(normalizeLogspace(logmarglik + np.log(mixprior)))
alphaPost = np.zeros_like(alphaPrior)
for z in range(M):
    alphaPost[z, :] = alphaPrior[z, :] + dataSS

grid = np.arange(0.0001, 0.9999, 0.01)
post = evalpdf(grid, mixpost, alphaPost)
prior = evalpdf(grid, mixprior, alphaPrior)
fig, axs = plt.subplots(1, 1)
fig.suptitle('mixBetaDemo', fontsize=10)
axs.plot(grid, prior, '--r', label='prior')
axs.plot(grid, post, '-b', label='posterior')
axs.legend()
fig.savefig('../figures/mixBetaDemo.png')

pbiased = 0
for k in range(M):
    a = alphaPost[k, 0]
    b = alphaPost[k, 1]
    pbiased += mixpost[k] * (1 - betainc(0.5, a, b))

pbiasedSimple = 1 - betainc(0.5, alphaPost[0, 0], alphaPost[0, 1])
