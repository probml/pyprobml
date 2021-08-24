# Approximate 2d posterior using pyro SVI
# https://www.ritchievink.com/blog/2019/06/10/bayesian-inference-how-we-are-able-to-chase-the-posterior/
# We use the same data and model as in posteriorGrid2d.py

import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


import pyro
import pyro.distributions as dist
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
import torch
import torch.distributions.constraints as constraints
import numpy as np

figdir = "../figures"
import os
def save_fig(fname):
    if figdir: plt.savefig(os.path.join(figdir, fname))

np.random.seed(0)

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
plt.show()

def model():
    # priors
    mu = pyro.sample('mu', dist.Normal(loc=torch.tensor(200.), 
                                       scale=torch.tensor(15.)))
    sigma = pyro.sample('sigma', dist.HalfCauchy(scale=torch.tensor(10.)))
    
    # likelihood
    with pyro.plate('plate', size=2):
        pyro.sample(f'obs', dist.Normal(loc=mu, scale=sigma), 
                    obs=torch.tensor([195., 185.]))
    
def guide():
    # variational parameters
    var_mu = pyro.param('var_mu', torch.tensor(180.))
    var_mu_sig = pyro.param('var_mu_sig', torch.tensor(5.), 
                             constraint=constraints.positive)
    var_sig = pyro.param('var_sig', torch.tensor(5.))
    
    # factorized distribution
    pyro.sample('mu', dist.Normal(loc=var_mu, scale=var_mu_sig))
    pyro.sample('sigma', dist.Chi2(var_sig))
    
pyro.clear_param_store()
pyro.enable_validation(True)

svi = SVI(model, guide, 
          optim=pyro.optim.ClippedAdam({"lr":0.01}), 
          loss=Trace_ELBO())

# do gradient steps
c = 0
for step in range(1000):
    c += 1
    loss = svi.step()
    if step % 100 == 0:
        print("[iteration {:>4}] loss: {:.4f}".format(c, loss))
        

sigma = dist.Chi2(pyro.param('var_sig')).sample((10000,)).numpy()
mu = dist.Normal(pyro.param('var_mu'), pyro.param('var_mu_sig')).sample((10000,)).numpy()

plt.figure()
plt.scatter(mu, sigma, alpha=0.01)
plt.xlim([extent[0], extent[1]])
plt.ylim([extent[2], extent[3]])
plt.ylabel('$\sigma$')
plt.xlabel('$\mu$')
plt.title('VI samples')
save_fig('bayes_unigauss_2d_pyro_post.pdf')
plt.show()