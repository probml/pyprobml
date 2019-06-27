

# Plots loss functions of form |x|**q


import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from scipy.stats import t, laplace, norm

x = np.linspace(-4, 4, 100)
plt.title('|x|^0.2')
plt.plot(x, np.absolute(x)**.2)
save_fig('lossFnQ2.pdf')

plt.figure()
plt.title('|x|')
plt.plot(x, np.absolute(x))
save_fig('lossFnQ10.pdf')

plt.figure()
plt.title('|x|^2')
plt.plot(x, np.absolute(x)**2)
save_fig('lossFnQ20.pdf')
plt.show()
