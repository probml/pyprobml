import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


np.random.seed(1)

p_default = ['cgatacggggtcgaa', 'caatccgagatcgca', 'caatccgtgttggga', 'caatcggcatgcggg', 'cgagccgcgtacgaa',
             'catacggagcacgaa', 'taatccgggcatgta', 'cgagccgagtacaga', 'ccatccgcgtaagca', 'ggatacgagatgaca']
np_type = np.float64

def conservationWeights(S, ssCorr=True):
    K = 4
    W = 0
    if (S.dtype == np_type):
        ssCorr = False
        W = np.divide(S, np.max(np.sum(S)))
    else:
        S = np.char.upper(S)
        ns, np = (S).shape
        U = np.unique(S)
        m = (U).size
        W = np.zeros((m, np))
        for j in range(0, m):
            W[j, :] = np.sum(S == U[j])
            W = W/ns

    F = W
    F[F == 0] = 1
    Sb = np.log2(K)
    Sf = -np.sum(np.multiply(np.log2(F), F), 1)
    E = 0
    if ssCorr:
        E = (K-1)/(2*np.log(2)*ns)
    else:
        E = 0
    R = Sb - Sf - E
    W = R*W
    return W


def seqlogoPmtk(p=p_default, ssCorr=True):
    W = np.max(conservationWeights(p, ssCorr), 0)
    #print(not(p.dtype == np.float64))


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
matrixEntropy = -np.sum(np.multiply(tmp, np.log2(tmp)), axis=1)
seqlogoPmtk(thetaHat)
