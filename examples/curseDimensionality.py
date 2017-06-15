#!/usr/bin/env python

# Show the curse of dimensionality.

import matplotlib.pyplot as pl
import numpy as np

ds = [1., 3., 5., 7., 10.]  # element is float, to make python2 work
s = np.linspace(0, 1, 100)
for d in ds:
  y = s ** (1 / d)
  pl.plot(s, y, 'b-')
  pl.text(0.3, 0.3**(1/d), 'd=%d' % d)
  pl.xlabel('Fraction of data in neighborhood')
  pl.ylabel('Edge length of cube')

pl.savefig('curseDimensionality.png')
pl.show()
