

import superimport

import numpy as np
from scipy import stats
from pyitlib import discrete_random_variable as drv

np.random.seed(42)

nzvals = 5
zvals = np.arange(nzvals)
pz = np.random.rand(nzvals); pz = pz / np.sum(pz)
distZ = stats.rv_discrete(values=(zvals, pz))

punif = np.ones(nzvals) / nzvals
distU = stats.rv_discrete(values=(zvals, punif))

xvals = zvals
nxvals = len(xvals)
pZtoX = np.zeros((nzvals, nzvals))
for z in zvals:
    prow = np.zeros(nzvals)
    p = 0.8
    q = (1-p)/2
    if z > 0:
        prow[z-1] = q
    prow[z] = p
    if z < nzvals-1:
        prow[z+1]= q
    prow = prow / np.sum(prow)
    pZtoX[z, :] = prow


def translate(source, trans_dist):
    n = len(source)
    target = np.zeros(n, dtype=np.int)
    for i in range(n):
        z = source[i]
        ptrans = trans_dist[z,:]
        dist = stats.rv_discrete(values=(xvals, ptrans))
        x = dist.rvs(size=1)
        target[i] = x
    return target


def generate_data(npatches, patch_size):
    U = np.zeros((npatches, patch_size), dtype=np.int)
    for i in range(npatches):
        U[i,:] = distU.rvs(size=patch_size)
    Z = distZ.rvs(size=patch_size)
    L = np.random.randint(0, npatches)
    source = U
    source[L, :] = Z
    X = np.zeros((npatches, patch_size), dtype=np.int)
    for i in range(npatches):
        X[i,:] = translate(source[i,:], pZtoX)
    return (X, L, Z, U)

def detect(X, M):
    npatches = np.size(X,0)
    score = np.zeros(npatches)
    for i in range(npatches):
        score[i] = drv.information_mutual(X[i,:], M)
    L = np.argmax(score)
    return (L, score)

npatches = 6
patch_size = 20
(X, L, Z, U) = generate_data(npatches, patch_size)
print('true L {}'.format(L))

#Msample = Z
counts = np.bincount(X[L,:], minlength=nxvals)
Mprob = counts/np.sum(counts)
Mdist = stats.rv_discrete(values=(xvals, Mprob))
Msample = Mdist.rvs(size=patch_size)

locn, scores = detect(X, Msample)
print('est L {}'.format(locn))
print('scores\n{}'.format(scores))
print('Z\n{}'.format(Z))
print('U\n{}'.format(U))
print('X\n{}'.format(X))
print('Mdist\n{}'.format(Mprob))
print('Msample\n{}'.format(Msample))
    
'''
nsamples = 25
Z1 = distZ.rvs(size=nsamples)
U1 = distU.rvs(size=nsamples)

X1 = translate(Z1, pZtoX)
X2 = translate(U1, pZtoX)

print('Z1\n{}'.format(Z1))
print('U1\n{}'.format(U1))

print('X1\n{}'.format(X1))
print('X2\n{}'.format(X2))

IZ1_X1 = drv.information_mutual(Z1, X1)
IZ1_X2 = drv.information_mutual(Z1, X2)
print('I(Z1,X1)={}, I(Z1, X2)={}'.format(IZ1_X1, IZ1_X2))
'''