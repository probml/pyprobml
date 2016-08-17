#!/usr/bin/env python

# Plots Negative Binomial distribution, using a custom
# negative binomial distribution.

import matplotlib.pyplot as pl
import numpy as np
import sympy
from scipy.stats import binom
from scipy.special import gammaln

# We use our own version of negative binomial here for two reasons:
# 1) scipy.stats.negbinom does not support real valued failures 
#      i.e scipy.stats.negbinom.pmf(x, f, p) only supports f an integer.
# 2) scipy.stats.negbinom does not handle arbitrary precision.
def _logNegBinom(x, f, p):
  # Note gammaln(x) = ln((x-1)!), so we add 1 to all terms inside gammaln.
  return gammaln(x+f) - gammaln(x+1) - gammaln(f) + f*np.log(1-p) + x*np.log(p)

logNegBinom = np.vectorize(_logNegBinom, excluded=['n', 'p'])

# Here sympy.S is used, to symbolically exponentiate,
# so that we don't lose any precision.
exp = np.vectorize(lambda x: sympy.exp(sympy.S(x)))
x = np.arange(100)
mu = 50

ps = [0.1, 0.5, 0.9]
styles = ['r:', 'k-.', 'g--']
fs = [1 + (1-p)/p * mu for p in ps]
labels = ['NB({:.2f}, {:.1f})'.format(f,p) for f,p in zip(fs,ps)]

pl.plot(x, binom.pmf(x, 100, .5), 'b-', label='binom(100, 0.5)')

for i,p in enumerate(ps):
  log_probabilities = logNegBinom(x, fs[i], p)
  pl.plot(x, exp(log_probabilities), styles[i], label=labels[i])

pl.legend()
pl.savefig('negBinomDemo_1.png')
pl.show()

fs = [1, 10, 30, 50]
styles = ['b-', 'r:', 'k-.', 'g--']
labels = ['NB({:.1f}, 0.5)'.format(n) for n in fs]

for i,n in enumerate(fs):
  log_probabilities = logNegBinom(x, fs[i], 0.5)
  pl.plot(x, exp(log_probabilities), styles[i], label=labels[i])
pl.axis([0, 100, 0, 0.25])
pl.legend()
pl.savefig('negBinomDemo_2.png')
pl.show() 
