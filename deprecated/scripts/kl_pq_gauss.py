# Plot KL(p,q) and KL(q,p) for 2d Gaussian
# Author: Drishtii
# Based on matlab code by Kevin Murphy
# https://github.com/probml/pmtk3/blob/master/demos/KLpqGauss.m

import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math
import pyprobml_utils as pml


def mnv_func(mu, Sigma):
    x1 = np.linspace(-1, 1, 201).reshape(-1, 1)
    x2 = np.linspace(-1, 1, 201).reshape(-1, 1)
    n1 = len(x1)
    n2 = len(x2)
    f = np.zeros((n1, n2))

    for i in range(n1):
        x = np.full((n2, 1), x1[i])
        q = np.concatenate((x, x2), axis=1)
        f[i, :] = multivariate_normal.pdf(q, mean=mu, cov=Sigma)

    return f


mu = np.array([0, 0])
Sigma = np.array([[1, 0.97], [0.97, 1]])
f = mnv_func(mu, Sigma)

mu = np.array([0, 0])
Sigma_kla = np.eye(2) / 25
klqp = mnv_func(mu, Sigma_kla)

mu = np.array([0, 0])
Sigma_klb = np.eye(2)
klpq = mnv_func(mu, Sigma_klb)

x1 = np.linspace(-1, 1, 201).reshape(-1, 1)
x2 = np.linspace(-1, 1, 201).reshape(-1, 1)
x1, x2 = np.meshgrid(x1, x2)

plt.contour(x1, x2, klpq, colors='r')
plt.contour(x1, x2, f, colors='b')
pml.savefig("KL_pq_gauss.pdf")
plt.show()

plt.contour(x1, x2, klqp, colors='r')
plt.contour(x1, x2, f, colors='b')
pml.savefig("KL_qp_gauss.pdf")
plt.show()
