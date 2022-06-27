# This function computes and returns MLE for the precision matrix of a gaussian_graphical_model(ggm)
# given known zeros in the adjacency matrix of the graph.
# Code in this file is based on https://github.com/probml/pmtk3/blob/master/toolbox/GraphicalModels/ggm/sub/ggmFitHtf.m

import superimport

import numpy as np


def ggm_fit_htf(S, G, max_iter):
    p = len(S)
    W = S
    prec_mat = np.zeros((p, p))
    beta = np.zeros((p - 1, 1))
    iters = 0
    converged = False
    norm_w = np.linalg.norm(W, 2)

    def converge_test(val, prev_val):
        threshold = 1e-4
        delta = abs(val - prev_val)
        avg = (abs(val) + abs(prev_val) + np.finfo(float).eps) / 2
        return (delta / avg) < threshold

    while not converged:
        for i in range(p):

            # partition W & S for i
            noti = [j for j in range(p) if j != i]
            W11 = W[np.ix_(noti, noti)]
            w12 = W[noti, i]
            s22 = S[i, i]
            s12 = S[noti, i]

            # find G's non-zero index in W11
            idx = np.argwhere(G[noti, i]).reshape(-1)  # non-zeros in G11
            beta[:] = 0
            beta[idx] = np.linalg.lstsq(W11[np.ix_(idx, idx)], s12[idx], rcond=-1)[0].reshape(-1, 1)

            # update W
            w12 = (W11 @ beta).reshape(-1)
            W[noti, i] = w12
            W[i, noti] = w12.T

            # update prec_mat (technically only needed on last iteration)
            p22 = max(0, 1 / (s22 - w12.T @ beta))  # must be non-neg
            p12 = (-beta * p22)
            prec_mat[noti, i] = p12.reshape(-1)
            prec_mat[i, noti] = p12.T
            prec_mat[i, i] = p22

        iters += 1
        converged = converge_test(np.linalg.norm(W, 2), norm_w) or (iters >= max_iter)
        norm_w = np.linalg.norm(W, 2)

    # ensure symmetry
    prec_mat = (prec_mat + prec_mat.T) / 2
    return prec_mat
