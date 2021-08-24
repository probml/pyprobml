# Show the curse of dimensionality.

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

ds = [1., 3., 5., 7., 10.]  
s = np.linspace(0, 1, 100)
for d in ds:
  y = s ** (1 / d)
  plt.plot(s, y, 'b-')
  plt.text(0.3, 0.3**(1/d), 'd=%d' % d)
  plt.xlabel('Fraction of data in neighborhood')
  plt.ylabel('Edge length of cube')

pml.savefig('curseDimensionality.pdf')
plt.show()
