# 1d grid approixmation to beta binomial model
# https://github.com/aloctavodia/BAP

import superimport

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pyprobml_utils as pml

def posterior_grid(heads, tails, grid_points=100):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)  # uniform prior
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (10, 3))
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(h, t, 20) 

plt.stem(grid, posterior, use_line_collection=True)
plt.title('grid approximation')
plt.yticks([])
plt.xlabel('θ')
pml.savefig('bb_grid.pdf')

plt.figure()
x = np.linspace(0, 1, 100)
xs = x #grid
post_exact = stats.beta.pdf(xs, h+1, t+1)
post_exact = post_exact / np.sum(post_exact)
plt.plot(xs, post_exact)
plt.title('exact posterior')
pml.savefig('bb_exact.pdf')

plt.show()