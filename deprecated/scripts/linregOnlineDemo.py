import superimport

from numpy import linalg
import numpy as np
from matplotlib import pyplot as plt
import pyprobml_utils as pml

def linregUpdateSS(ss, xnew, ynew, xAll, yAll):
    if len(ss.keys()) == 0:
        ss['xbar'] = xnew
        ss['ybar'] = ynew
        ss['Cxx'] = 0.0
        ss['Cxy'] = 0.0
        ss['Cxx2'] = 0.0
        ss['Cxy2'] = 0.0
        ss['Cxx3'] = 0.0
        ss['Cxy3'] = 0.0
        ss['n'] = 1

        return np.array([0.0, 0.0]), ss

    else:

        ssOld = ss.copy()
        n = ss['n']
        n1 = n + 1
        ss['n'] = ss['n'] + 1

        ss['xbar'] = ssOld['xbar'] + (1.0 / n1) * (xnew - ssOld['xbar'])

        ss['ybar'] = ssOld['ybar'] + (1.0 / n1) * (ynew - ssOld['ybar'])

        ss['Cxy'] = (1.0 / n1) * (
                xnew * ynew + n * ssOld['Cxy'] + n * ssOld['xbar'] * ssOld['ybar'] - n1 * ss['xbar'] * ss['ybar'])

        ss['Cxx'] = (1.0 / n1) * (
                xnew ** 2 + n * ssOld['Cxx'] + n * ssOld['xbar'] * ssOld['xbar'] - n1 * ss['xbar'] * ss['xbar'])

        ndx = np.arange(ss['n'])

        assert (np.allclose(ss['xbar'], np.mean(xAll[ndx])))
        assert (np.allclose(ss['ybar'], np.mean(yAll[ndx])))
        assert (np.allclose(ss['Cxy'], np.mean((xAll[ndx] - ss['xbar']) * (yAll[ndx] - ss['ybar']))))
        assert (np.allclose(ss['Cxx'], np.mean((xAll[ndx] - ss['xbar']) ** 2)))

    w1 = ss['Cxy'] / ss['Cxx']
    w0 = ss['ybar'] - w1 * ss['xbar']
    w = np.array([w0, w1])

    ndx = np.arange(ss['n'])
    ww = np.dot(linalg.pinv(addOnes(xAll[ndx])), yAll[ndx])
    assert (np.allclose(ww, w))
    return w, ss


def polyDataMake():
    '''
    sampling : thibaux
    '''
    xtrain = np.linspace(0, 20, 21)
    xtest = np.arange(0, 20, 0.1)
    sigma2 = 4
    w = np.array([-1.5, 1 / 9.0])
    ytrain = w[0] * xtrain + w[1] * xtrain ** 2 + np.random.standard_normal(len(xtrain)) * np.sqrt(sigma2)
    ytestNoisefree = w[0] * xtest + w[1] * xtest ** 2
    return xtrain, ytrain, xtest, ytestNoisefree


def addOnes(X):
    X = X[..., np.newaxis]
    X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
    return X

standardize = False

np.random.seed(3)

xtrain, ytrain, _, _ = polyDataMake()

if standardize:
    xtrain = (xtrain - xtrain.mean(axis=0)) / xtrain.std(axis=0)

N = len(xtrain)
wBatch = np.dot(linalg.pinv(addOnes(xtrain)), ytrain)

ss = {}
w = np.zeros((2, N))
for i in range(N):
    w[:, i], ss = linregUpdateSS(ss, xtrain[i], ytrain[i], xtrain, ytrain)

fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
ax1.plot(np.arange(2, N + 1), w[0, 1:], color='#e41a1c', marker='o', linestyle='None', linewidth=2, label='w0')
ax1.plot(np.arange(2, N + 1), w[1, 1:], color='#377eb8', marker='*', linestyle='None', linewidth=2, label='w1')
ax1.plot(np.arange(1, N + 1), wBatch[0] * np.ones(N), color='#e41a1c', linestyle='-', linewidth=2, label='w0 batch')
ax1.plot(np.arange(1, N + 1), wBatch[1] * np.ones(N), color='#377eb8', linestyle=':', linewidth=2, label='w1 batch');
ax1.legend()
ax1.set_title('linregOnlineDemo')
ax1.set_ylabel('weights')
ax1.set_xlabel('time')
pml.savefig('linregOnlineDemo.pdf')
plt.show()