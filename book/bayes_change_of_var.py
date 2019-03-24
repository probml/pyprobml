

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from scipy.stats import norm


def ginv(x):
    """transform func"""
    return 1 / (1 + np.exp(-x + 5))

mu, sigma = 6, 1
n = 10 ** 6
x = norm.rvs(size=n, loc=mu, scale=sigma)
x_range = np.arange(0, 10, 0.01)
#plot the histogram
hist, bin_edges = np.histogram(x, bins=50, normed=True)
pl.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0], color='r')
hist, bin_edges = np.histogram(ginv(x), bins=50, normed=True)
plt.barh(bin_edges[:-1], hist, height=bin_edges[1] - bin_edges[0], color='g')

#plot transform function
plt.plot(x_range, ginv(x_range), 'b', lw=5)

#plot line at mu
plt.plot([mu, mu], [0, ginv(mu)], 'y', lw=5)
plt.plot([0, mu], [ginv(mu), ginv(mu)], 'y', lw=5)

save_fig('bayesChangeOfVar.pdf')
plt.show()
