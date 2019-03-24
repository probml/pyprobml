#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:05:32 2019

@author: kpmurphy
"""

import numpy as np
from scipy import stats
probs = [0.3, 0.5, 0.2]
rv_d = stats.multinomial(1, probs)
n = 10
samples_d = rv_d.rvs(size=n) #np.random.multinomial(1, probs, n) # n x 3 one hots
ll_d = rv_d.logpmf(samples_d)
rv_c = stats.norm(0,1)
samples_c = rv_c.rvs(size=n)
ll_c = rv_c.logpdf(samples_c)


#ll_d2 = logprob(rv_d, samples_d)
#assert np.allclose(ll_d, ll_d2)
#
#ll_c2 = logprob(rv_c, samples_c)
#assert np.allclose(ll_c, ll_c2)

import abc
class ProbDist(metaclass=abc.ABCMeta):
    """ Abstract base class for probability distributions.
    
    This is a lightweight wrapper around some scipy.stats distributions.
    This class provides logprob(), which works for both discrete and
    continuous distributions, instead of needing to use logpmf or logpdf.
    It also provides fit() for discrete and multivariate distributions,
    which scipy is missing. Furthermore, the fit
    method optionally implements MAP estimation as well as MLE.
    The sample() interface is also simplified.
    """
 
    def logprob(self, x):
      """Returns n vector of log probabilities for each row of x."""
      if not(hasattr(self.rv, "logpdf")):
        # Discrete distributions do not have a logpdf method.
        return self.rv.logpmf(x)
      else:
        # Note that continuous distributions have the logpmf method
        # (but fail to implement it), as well as logpdf.
        return self.rv.logpdf(x)
  
    def sample(self, n):
      """Return n*d matrix of samples."""
      return self.rv.rvs(size=n)

    def fit(self, x):
      """Return MLE (or MAP estimate) for parameters given samples in rows of x"""
      raise NotImplementedError()
      
    def entropy(self):
      """Returns (differential) entropy."""
      raise self.rv.entropy()
      
    def mean(self):
      """Returns d vector containing mean."""
      raise self.rv.mean()
      
    def var(self):
      """Returns d vector containing marginal variances."""
      raise self.rv.var()
      
    
class CategoricalProbDist(ProbDist):
  def __init__(self, probs):
    """Create categorical distribution with c=length(probs) classes."""
    self.probs = probs
    self.rv = stats.multinomial(1, probs)
    
  def fit(self, x, dirichlet_prior=1.0):
    """Compute MAP for the parameters using Dirichlet prior.
    The MAP estimate is given by
      theta(c) = (N(c) + alpha(c) - 1.0) / (N + alpha)
    where N(c) = number of times state c occurs in x,
    alpha(c) = pseudocount for state c,
    N = sum_c N(c) is the sample size (num rows of x),
    alpha = sum_c alpha(c) is the prior strength.
    By default, we use a uniform prior of alpha(c) = 1,
    which corresponds to MLE."""
    pass
      

class UniGaussProbDist(ProbDist):
  def __init__(self, mean, var):
    """We provide the mean and *variance* (not std) to be consistent
    with the multivariate case."""
    self._mean = mean
    self._std = np.sqrt(var)
    self.rv = stats.norm(self._mean, self._std)
    
  
  def fit(self, x):
    """Compute MLE for the parameters. Note that the estimate
    for the variance is the MLE, not the unbiased estimate."""
    # The scipy fit method uses MLE, not unbiased estimator.
    # Just to be safe, we implement the equations explicitly.
    #params = stats.norm.fit(x)
    #self.__init(params[0], params[1])
    n = np.shape(x)[0]
    m = np.mean(x)
    ss = np.sum(np.power(x, 2))
    self.__init(m, ss/n - m**2)


class LaplaceProbDist(ProbDist):
  def __init__(self, loc, scale):
    self._loc = loc
    self._scale = scale
    self.rv = stats.laplace(loc, scale)
    
  def fit(self, x):
    """Compute MLE for the parameters."""
    # MLE for mean is the median, 
    # https://math.stackexchange.com/questions/240496/finding-the-maximum-likelihood-estimator
    params = stats.laplace.fit(x)
    self.__init(params[0], params[1])

class MultiGaussProbDist(ProbDist):
  def __init__(self, mean, cov):
    self._mean = mean
    self._cov = cov
    self.rv = stats.multivariate_normal(self._mean, self._cov)
    
  def fit(self, X, Nprior=1):
    """Compute MLE for mean and MAP estimate for covariance from rows of X.
    The prior for the covariance is a Wishart with mode I/Nprior, where Nprior
    is the effect strength of the prior.
    """
    self._mean = np.mean(X, axis=0) 
    N, D = np.shape(X)
    S = np.cov(X, rowvar=0, ddof=1) * N # divide by N, not N-1
    Sprior = np.eye(D)
    self._cov = (S + Sprior)/(N + Nprior)
    
 #https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html 
  
probs = [0.4, 0.2, 0.3]
dist1 = CategoricalProbDist(probs)
xs1 = dist1.sample(n)
ll1 = dist1.logprob(xs1)

m = 0.1; s = 2.0;
dist2 = UniGaussProbDist(m, s)
xs2 = dist2.sample(n)
ll2 = dist2.logprob(xs2)
assert np.isclose(dist2.mean(), m)
assert np.isclose(dist2.var(), s**2)

m = 0.1; s = 2.0;
dist3 = LaplaceProbDist(m, s)
xs3 = dist3.sample(n)
ll3 = dist3.logprob(xs3)
assert np.isclose(dist3.mean(), m)
assert np.isclose(dist3.var(), 2*s**2)
