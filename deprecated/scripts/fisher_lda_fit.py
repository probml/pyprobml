'''
This file implements fisher projection of classification data onto K(< no.of.classes) dimensions
to further fit it with LDA.
Referenced from
https://github.com/probml/pmtk3/blob/master/toolbox/SupervisedModels/fisherLda/fisherLdaFit.m
Author: Srikar-Reddy-Jilugu(@always-newbie161)
'''
import superimport

import numpy as np


def fisher_lda_fit(X, y, kdims):
    """
    :param X: shape(nsamples, ndims)
    :param y: shape(nsamples, 1)
    :param kdims: int
    :return W: Linear Projection Matrix; ndarray of shape(ndims, kdims)
    """
    nclasses = np.max(y)  # no .of.classes
    nsamples, ndims = X.shape

    if nclasses == 2:
        # assuming y is from {1,2}
        ndx1 = np.where(y == 1)[0]
        ndx2 = np.where(y == 2)[0]
        mu1, mu2 = np.mean(X[ndx1, :]), np.mean(X[ndx2, :])
        S1, S2 = np.cov(X[ndx1, :]), np.cov(X[ndx2, :])
        Sw = S1 + S2
        W = np.linalg.inv(Sw) @ (mu2 - mu1)
    else:
        # assuming y is from {1,2,..nclasses}
        muC = np.zeros((nclasses, ndims))
        for c in range(0, nclasses):
            ndx = np.where(y == (c + 1))[0]
            muC[c, :] = (np.mean((X[ndx, :]), axis=0))

        mu_matrix = np.squeeze(muC[y-1, :], axis=1)
        Sw = (X - mu_matrix).T @ (X - mu_matrix)
        muX = np.mean(X, axis=0)
        Sb = (np.ones((nclasses, 1))*muX - muC).T @ (np.ones((nclasses, 1))*muX - muC)
        _, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
        W = eigvecs[:, :kdims]

    return W
