# -*- coding: utf-8 -*-
"""
Author : Ang Ming Liang

This code is based on https://github.com/probml/pmtk3/blob/master/demos/sliceSamplingDemo1d.m
"""

import superimport

import numpy as np
import matplotlib.pyplot as plt
from mcmc_utils import slice_sample
import pyprobml_utils as pml

seed = 123
rng = np.random.default_rng(seed)

def pdf(x):
   return np.exp(-x**2/2)*(1+(np.sin(3*x))**2)*(1+(np.cos(5*x))**2)
def logpdf(x):
  return np.log(pdf(x))

x = np.array([1.])
out = slice_sample(x, logpdf, iters=4000, sigma=5, burnin=1000, rng=rng)

fig, ax = plt.subplots()

# Set dim of the figure
ax.set_xlim(-5, 4)
ax.set_ylim(0, 180)

# Plt histogram
bin = ax.hist(out[0], bins=75, ec="black",lw=0.5)

xd = bin[1] # Get bins size
binwidth = xd[1]-xd[0] # Finds the width of each bin

y = 563.98*binwidth*pdf(np.linspace(-5,4,1000));
ax.plot(np.linspace(-5,4,1000),y,'r')
pml.savefig('sliceSamplingDemo1d.pdf')
plt.show()
