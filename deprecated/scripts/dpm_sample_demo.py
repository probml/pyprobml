# Sample from a DP mixture of 2D Gaussians
# Converted from https://github.com/probml/pmtk3/blob/master/demos/dpmSampleDemo.m

import superimport

import pyprobml_utils as pml

import numpy as np
import scipy as sp
from scipy import linalg
from scipy import stats

import matplotlib.pyplot as plt

seeds = [2, 3]
aa = 2  # alpha
nn = 1000  # number of data points
sigma = 1*np.eye(2)  # mean covariance matrix
vsigma = 1
dof = 10  # degree of freedom
mu = np.zeros((2, 1))  # mean of means
mv = 8*np.ones((2, 1))  # std of means
ax = 30

for trial, seed in enumerate(seeds):
    np.random.seed(seed)
    # Sample from CRP prior
    T = []
    zz = np.zeros((2, nn)).flatten()
    for ii in range(nn):
        pp = np.array(T+[aa])
        kk = np.sum(np.random.rand(1)*np.sum(pp) > np.cumsum(pp))
        if kk < len(T):
            T[kk] += 1
        else:
            T += [0 for _ in range(kk-len(T)+1)]
            T[kk] = 1
        zz[ii] = kk

    # Generate random parameters for each mixture component
    mm = np.zeros((2, len(T)))
    vv = np.zeros((2, 2, len(T)))
    for kk in range(len(T)):
        mm[:, [kk]] = (np.random.randn(2, 1)*mv+mu)
        vv[:, :, kk] = sp.linalg.sqrtm(sp.stats.wishart(
            df=dof, scale=sigma).rvs(1)) * np.sqrt(np.random.gamma(vsigma, 1))

    # Generate data from each component
    xx = np.zeros((2, nn))
    for ii in range(nn):
        kk = int(zz[ii])
        xx[:, [ii]] = (vv[:, :, kk].dot(
            np.random.randn(2, 1)) + mm[:, [kk]])

    # Plot
    bb = np.arange(0, 2*np.pi, .02)
    ss = [50, 200, 500, 1000]
    plt.figure()
    for jj, sj in enumerate(ss):
        hh, _ = np.histogram(zz[:sj], np.arange(0, max(zz[:sj])))
        cc = np.where(hh >= 1)[0]
        plt.plot(xx[0, :sj], xx[1, :sj], '.', markersize=7)
        for kk in list(cc):
            uu = vv[:, :, kk]
            circ = mm[:, [kk]].dot(np.ones((1, len(bb)))) + \
                uu.dot(np.vstack([np.sin(bb), np.cos(bb)]))
            plt.plot(circ[0, :], circ[1, :], linewidth=2, color='k')
            plt.xlim(-ax, ax)
            plt.ylim(-ax, ax)
            plt.xticks([])
            plt.yticks([])
            N = sj
        pml.savefig("dpmSampleSeed%sN%s.pdf" % (seed, N))
        plt.show()
