#!/usr/bin/env python

# Plot the softmax function.

import matplotlib.pyplot as pl
import numpy as np

def softmax(a, t):
  e = np.exp((1.0 * np.array(a)) / t)
  return e / np.sum(e)

T = [100, 5, 1]
a = [3, 0, 1];
ind = [1, 2, 3]

for i in range(len(T)):
  pl.bar(ind, softmax(a, T[i]))
  pl.ylim(0, 1)
  pl.title('T = %d' % T[i])
  pl.savefig('softmax_temp%d.png' % T[i])
  pl.show()
