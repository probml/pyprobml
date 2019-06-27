# Plots L2, L1 and Huber losses.

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


delta = 1.5
huber = lambda x: (x**2/2) * (x <= delta) + (delta * abs(x) - delta**2/2) * (x > delta)
l2 = lambda x: abs(x)**2
l1 = abs

funs = [l2, l1, huber]
styles = ['r-', 'b:', 'g-.']
labels = ['l2', '11', 'huber']
x = np.arange(-3, 3, .01)

for i, fun in enumerate(funs):
  plt.plot(x, fun(x), styles[i], label=labels[i])

plt.axis([-3, 3, -0.5, 5])
plt.legend()
save_fig(os.path.join(figdir, 'huberLossPlot.pdf'))
plt.show()
