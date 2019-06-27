# Bimodal distribution (mixture of two 1d Gaussians)
# Based on https://github.com/probml/pmtk3/blob/master/demos/bimodalDemo.m


import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from scipy.stats import norm

# Define two normal distrubutions and their corresponding weights.
mu = [0, 2]
sigma = [1, 0.05]
n = [norm(loc=mu[i], scale=sigma[i]) for i in range(2)]
w = [0.5, 0.5]

# Define a set of x points for graphing.
xs = np.linspace(-2, 2*mu[1], 600)

# Combine the two distributions by their weights, evaluated at the x points.
p = sum(w[i] * n[i].pdf(xs) for i in range(2))

# Calculate the mean of the final distribution.
mean_p = np.mean(xs * p)

# Plot the final distribution and its mean.
linewidth = 3
plt.figure()
plt.plot(xs, p, 'black', linewidth=linewidth)
plt.vlines(mean_p, ymin=0, ymax=max(p), color='red', linewidth=linewidth)
save_fig('bimodalSpike.pdf')
plt.show()


# Another example, with two modes
mu = [0, 2]
sigma = [0.5, 0.5]
n = [norm(loc=mu[i], scale=sigma[i]) for i in range(2)]
w = [0.5, 0.5]
xs = np.linspace(-2, 2*mu[1], 600)
p = sum(w[i] * n[i].pdf(xs) for i in range(2))

plt.figure()
linewidth = 3
plt.plot(xs, p, 'black', linewidth=linewidth)
save_fig('bimodalDistribution.pdf')
plt.show()
