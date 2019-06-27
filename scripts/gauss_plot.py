# Plot pdf and cdf of standard normal

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from scipy.stats import norm


X = np.linspace(-3, 3, 500)
rv = norm(0, 1)
fig, ax = plt.subplots()
ax.plot(X, rv.pdf(X))
plt.title("Gaussian pdf")
save_fig("gaussian1d.pdf")
plt.show()

fig, ax = plt.subplots()
ax.plot(X, rv.cdf(X))
plt.title("Gaussian cdf")
save_fig("gaussianCdf.pdf")
plt.show()
