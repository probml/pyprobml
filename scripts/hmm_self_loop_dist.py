# Hidden Markov Model with self loops
# Author: Drishtii@
# Based on
# https://github.com/probml/pmtk3/blob/master/demos/hmmSelfLoopDist.m

import superimport

import pyprobml_utils as pml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import nbinom
import math

Ls = np.linspace(1, 600, 100).astype(int)     # path lengths
ns = np.array([1, 2, 5])                      # nos. of nodes
p = 0.99                                      # self-loop probability

def binomial(n, k):                         
  return math.log(math.factorial(n) // math.factorial(k) // math.factorial(n - k))

def plot_nbinom_dist(ns, Ls, p):
  logp = np.log(p) 
  logq = np.log(1-p)
  h = np.zeros((len(ns), 1))
  for i in range(len(ns)):
    n = ns[i]
    ps = np.zeros((len(Ls), 1))
    for j in range(len(Ls)):
      L=Ls[j]
      if (L>=n):
        ps[j] = np.exp(binomial(L-1, n-1) + (L-n)*logp + n*logq)
    plt.ylim(0, 0.013)
    plt.plot(Ls, ps)
    
# Negative Binomial Distribution:
plot_nbinom_dist(ns, Ls, p)
plt.legend(['n=1', 'n=2', 'n=5'])
pml.savefig('hmm_self_loop.pdf')
plt.show()