# Empirical Bayes for 8 schools

import superimport

import numpy as np

# Data of the Eight Schools Model
J = 8
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])

d = len(y);
mu = np.mean(y); # MLE-II
V    = np.sum(np.square(y-mu));
s2   = V/d;
sigma2 = np.mean(np.square(sigma))
tau2 = np.maximum(0, s2-sigma2); # approx

lam = sigma2/(sigma2 + tau2); 
print(lam)
muShrunk = mu + (1-lam)*(y-mu);
print(muShrunk) 