

# Show the curse of dimensionality.


import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


ds = [1., 3., 5., 7., 10.]  
s = np.linspace(0, 1, 100)
for d in ds:
  y = s ** (1 / d)
  plt.plot(s, y, 'b-')
  plt.text(0.3, 0.3**(1/d), 'd=%d' % d)
  plt.xlabel('Fraction of data in neighborhood')
  plt.ylabel('Edge length of cube')

save_fig('curseDimensionality.pdf')
plt.show()
