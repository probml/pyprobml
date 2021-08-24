# This file shows demo of robust prior(using cauchy prior)
# This code is based on https://github.com/probml/pmtk3/blob/master/demos/robustPriorDemo.m

import superimport

import numpy as np
import math
from scipy.stats import norm, cauchy
import scipy.integrate as integrate

obs_x = 5
obs_var = 1
obs_std = math.sqrt(obs_var)

# prior constraints:
# smooth, median(theta)=0, mode(theta)=1, theta can be (-inf,1),(-1,0),(0,1),(1,inf) with p=0.25

# taking a gaussian prior which satisfies the above constraints
prior_var = 2.19
prior_mu = 0
prior_std = math.sqrt(prior_var)
p_range = norm.cdf(1, prior_mu, prior_std) - norm.cdf(-1, prior_mu, prior_std)
assert np.allclose(p_range, 0.5, 1e-2)

# Computing posterior mean using gaussian prior
post_var = 1 / (1 / obs_var + 1 / prior_var)
post_mean = post_var * (prior_mu / prior_var + obs_x / obs_var)

assert np.allclose(post_mean, 3.43, 1e-2)

# taking a cauchy prior which satisfies the above constraints
loc = 0
scale = 1
p_range = cauchy.cdf(1, loc, scale) - cauchy.cdf(-1, loc, scale)
assert np.allclose(p_range, 0.5, 1e-2)

# Computing posterior mean using cauchy prior
inf = 5.2
lik = lambda theta: norm.pdf(obs_x, theta, obs_std)
prior = lambda theta: cauchy.pdf(obs_x, theta, obs_std)
post = lambda theta: lik(theta) * prior(theta)
Z = integrate.quad(post, -inf, inf)[0]
post_mean = integrate.quad(lambda theta: theta*post(theta)/Z, -inf, inf)[0]
assert np.allclose(post_mean, 4.56, 1e-2)
