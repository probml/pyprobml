
import superimport

import numpy as np
import matplotlib.pyplot as plt

def gaussProb(X, mu, Sigma):
    d = 1
    X = np.reshape(X, newshape=(-1, d), order='F')
    X = X - mu.T
    logp = -0.5 * np.sum(X*X/Sigma, axis=1)
    logZ = 0.5 * d * np.log(2*np.pi) + 0.5 * np.log(Sigma)
    logp = logp - logZ
    p = np.exp(logp)

    return p

k0s = np.arange(4)
xrange = np.arange(-1, 2.55, 0.05)
n=5
thetaTrue = 1
sigmaTrue = 1
thetaPrior = 0
colors = ['b', 'r', 'k', 'g', 'c', 'y', 'm', 'r', 'b', 'k', 'g', 'c', 'y', 'm']
styles = ['-', ':', '-.', '--', '-', ':', '-.', '--', '-', ':', '-.', '--', '-', ':', '-.', '--']

plt.figure(figsize=(12, 7))
names = []
for ki in range(len(k0s)):
    k0 = k0s[ki]
    w = n / (n + k0)
    v = w**2 * sigmaTrue**2 / n
    thetaEst = w*thetaTrue + (1-w)*thetaPrior
    names.append('postMean{0:02d}'.format(k0s[ki]))
    plt.plot(xrange, gaussProb(xrange, thetaEst, np.sqrt(v)), color=colors[ki], linestyle=styles[ki], linewidth=3)
plt.title('sampling distribution, truth = {}, prior = {}, n = {}'.format(thetaTrue, thetaPrior, n), fontweight="bold")
plt.legend(names)
plt.show()

ns = np.arange(1, 50, 2)
mseThetaE = np.zeros((len(ns), len(k0s)))
mseThetaB = np.zeros((len(ns), len(k0s)))

for ki in range(len(k0s)):
    k0 = k0s[ki]
    ws = ns / (ns + k0)
    mseThetaE[:, ki] = sigmaTrue**2/ns
    mseThetaB[:, ki] = ws**2 * sigmaTrue**2 / ns + (1-ws)**2 * (thetaPrior-thetaTrue)**2

ratio = mseThetaB / mseThetaE

plt.figure(figsize=(12, 7))
for ki in range(len(k0s)):
    plt.plot(ns, ratio[:, ki],  color=colors[ki], linestyle=styles[ki], linewidth=3)

plt.legend(names)
plt.ylabel('relative MSE');
plt.xlabel('sample size')
plt.title('MSE of postmean / MSE of MLE', fontweight="bold")
plt.tight_layout()
plt.show()