import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
import pyprobml_utils as pml

fs = 12
np.random.seed(0)
true_mu1 = -10
true_mu2 = 10
true_pi = 0.5


true_sigma = 5


n_obs = 100
obs = ([true_mu1 + true_sigma*np.random.randn(1, n_obs), true_mu2 + true_sigma*np.random.randn(1, n_obs)])
obs = np.reshape(obs, [1, 200])
obs = np.transpose(obs)

plt.figure()
histogram = plt.hist(obs)
pml.savefig('gmm_lik_surface_hist')
plt.show()
#print(obs.shape)

dmu = .5;
mu1_bins = np.arange(-20, 20+dmu, dmu)
mu2_bins = np.arange(-20, 20+dmu, dmu)

n_mu1_bins = mu1_bins.size
n_mu2_bins = mu2_bins.size

lik_bins = np.zeros((n_mu1_bins, n_mu2_bins))

for b1 in range(0, n_mu1_bins):
    for b2 in range(0, n_mu2_bins):
        p1 = true_pi * multivariate_normal.pdf(obs, mu1_bins[b1], true_sigma)
        p2 = (1-true_pi) * multivariate_normal.pdf(obs, mu1_bins[b2], true_sigma)
        lik_bins[b1, b2] = sum(np.log(p1 + p2));

plt.figure()
plt.contour(lik_bins)
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
pml.savefig('gmm_lik_surface_contour')
plt.show()