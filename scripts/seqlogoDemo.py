import numpy as np
import numpy.matlib

np.random.seed(1)


def dirichlet_sample(a, n=1):

    row = a.shape[0] == 1
    a = a.flatten()
    y = np.random.gamma(np.matlib.repmat(a, 1, n))
    r = np.sum(y, axis=1)
    r[np.argwhere(r == 0)] = 1
    r = np.divide(y, np.matlib.repmat(r, (y).shape[0], 1))
    if row:
        r = np.transpose(r)
    return r


def sampleDiscrete(prob, r=1, c=1):
    n = len(prob)
    R = np.random.rand(r, c)
    M = np.ones((r, c))
    cumprob = np.cumsum(prob[:])

    if n < r*c:
        for i in range(0, n-1):
            M = M + (R > cumprob[i])

    cumprob2 = cumprob[1: n-1]
    for i in range(0, r):
        for j in range(0, c):
            M[i, j] = sum(R[i, j] > cumprob2)+1

    return M


Nseq = 10
Nlocn = 15
Nletters = 4
Nmix = 4
pfg = 0.30

a = (pfg/Nmix*np.array([1]*Nmix)).reshape(1, 4)
b = np.array([1 - pfg]).reshape(1, 1)
mixweights = np.concatenate((a, b), axis=1)

z = sampleDiscrete(prob=mixweights, r=1, c=Nlocn)
alphas = 1*np.ones((Nletters, Nmix+1))

for i in range(0, Nmix):
    alphas[i, i] = 20

alphas[:, Nmix] = np.ones((Nletters, 1)).reshape(4, )

theta = np.zeros((Nletters, Nlocn))
data = np.zeros((Nseq, Nlocn))
chars = np.transpose(np.array(['a', 'c', 'g', 't', '-']))

dataStr = numpy.ndarray(data.shape, dtype=object)
# print(dataStr)
z = np.transpose(z)
for t in range(0, Nlocn):
    theta[:, t] = np.transpose(dirichlet_sample(
        alphas[:, z[t].astype('int')], 1)).reshape(4, )
    data[:, t] = sampleDiscrete(prob=theta[:, t], r=Nseq, c=1).reshape(10, )
    dataStr[:, t] = chars[data[:, t].astype('int')]
'''
for i in range(0, Nseq):
  for t in range(0, Nlocn):
    print(dataStr[i, t], end = " ")
  print()
'''
counts = np.zeros((4, Nlocn))
for c in range(0, 4):
    counts[c, :] = np.sum(data == c, axis=0)

thetaHat = counts/Nseq
tmp = thetaHat
tmp[tmp == 0] = 1
matrixEntropy = -np.sum(np.multiply(tmp, np.log2(tmp)), axis = 1)
print(matrixEntropy)