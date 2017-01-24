#!/usr/bin/env python

# Plot the bernoulli entropy.

import matplotlib.pyplot as pl
import numpy as np

def entropy(p):
    """calculate the entropy"""
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return h

x = np.linspace(0.01, 0.99, 100)
y = entropy(x)

pl.plot(x, y)
pl.xlabel('p(X=1)')
pl.ylabel('H(X)')
pl.savefig('bernoulliEntropyFig.png')
pl.show()
