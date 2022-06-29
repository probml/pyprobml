# Approximate 2d posterior using PyMc3
# https://www.ritchievink.com/blog/2019/06/10/bayesian-inference-how-we-are-able-to-chase-the-posterior/
# We use the same data and model as in posteriorGrid2d.py

import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymc3 as pm
import pyprobml_utils as pml
import os

data = np.array([195, 182])

# lets create a grid of our two parameters
mu = np.linspace(150, 250)
sigma = np.linspace(0, 15)[::-1]
mm, ss = np.meshgrid(mu, sigma)  # just broadcasted parameters
likelihood = stats.norm(mm, ss).pdf(data[0]) * stats.norm(mm, ss).pdf(data[1])
aspect = mm.max() / ss.max() / 3
extent = [mm.min(), mm.max(), ss.min(), ss.max()]
# extent = left right bottom top

prior = stats.norm(200, 15).pdf(mm) * stats.cauchy(0, 10).pdf(ss)
# Posterior - grid
unnormalized_posterior = prior * likelihood
posterior = unnormalized_posterior / np.nan_to_num(unnormalized_posterior).sum()

plt.figure()
plt.imshow(posterior, cmap='Blues', aspect=aspect, extent=extent)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.title('Grid approximation')
pml.savefig('bayes_unigauss_2d_grid.pdf')
plt.show()


with pm.Model():
    # priors
    mu = pm.Normal('mu', mu=200, sd=15)
    sigma = pm.HalfCauchy('sigma', 10)
    # likelihood
    observed = pm.Normal('observed', mu=mu, sd=sigma, observed=data)
    # sample
    trace = pm.sample(draws=10000, chains=2, cores=1)
    
pm.traceplot(trace);

plt.figure()
plt.scatter(trace['mu'], trace['sigma'], alpha=0.01)
plt.xlim([extent[0], extent[1]])
plt.ylim([extent[2], extent[3]])
plt.ylabel('$\sigma$')
plt.xlabel('$\mu$')
plt.title('MCMC samples')
pml.savefig('bayes_unigauss_2d_pymc3_post.pdf')
plt.show()
