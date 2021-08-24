# 2d grid posterior approximation to N(x|mu,sigma^2) N(mu) Cauchy(sigma)
# https://www.ritchievink.com/blog/2019/06/10/bayesian-inference-how-we-are-able-to-chase-the-posterior/

import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

figdir = "../figures"
import os

data = np.array([195, 182])

# lets create a grid of our two parameters
mu = np.linspace(150, 250)
sigma = np.linspace(0, 15)[::-1]

mm, ss = np.meshgrid(mu, sigma)  # just broadcasted parameters

# Likeliohood
likelihood = stats.norm(mm, ss).pdf(data[0]) * stats.norm(mm, ss).pdf(data[1])
aspect = mm.max() / ss.max() / 3
extent = [mm.min(), mm.max(), ss.min(), ss.max()]
# extent = left right bottom top

plt.figure()
plt.imshow(likelihood, cmap='Reds', aspect=aspect, extent=extent)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.title('Likelihood')
pml.savefig('posteriorGridLik.pdf')
plt.show()

# Prior
prior = stats.norm(200, 15).pdf(mm) * stats.cauchy(0, 10).pdf(ss)

plt.figure()
plt.imshow(prior, cmap='Greens', aspect=aspect, extent=extent)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.title('Prior')
pml.savefig('posteriorGridPrior.pdf')
plt.show()


# Posterior - grid
unnormalized_posterior = prior * likelihood
posterior = unnormalized_posterior / np.nan_to_num(unnormalized_posterior).sum()

plt.figure()
plt.imshow(posterior, cmap='Blues', aspect=aspect, extent=extent)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')
plt.title('Posterior')
pml.savefig('posteriorGridPost.pdf')
plt.show()
