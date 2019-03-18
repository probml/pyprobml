#https://docs.pymc.io/notebooks/api_quickstart.html
#%matplotlib inline
import numpy as np
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt
from time import time

#sns.set_context('notebook')
plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

np.random.seed(0)
N = 100
x = np.random.randn(100)
mu_prior = 1.1
sigma_prior = 1.2
Sigma_prior = sigma_prior**2
sigma_x = 1.3
Sigma_x = sigma_x**2
with pm.Model() as model:
    mu = pm.Normal('mu', mu=mu_prior, sd=sigma_prior)
    obs = pm.Normal('obs', mu=mu, sd=sigma_x, observed=x)
    time_start = time()
    mcmc_samples = pm.sample(1000, tune=500) # mcmc
    print('time spent MCMC {:0.3f}'.format(time() - time_start))
    time_start = time()
    vi_post = pm.fit() # variational inference
    print('time spent VI {:0.3f}'.format(time() - time_start))
    vi_samples = vi_post.sample(1000)
    
mu_clamped = -0.5    
logp = model.logp({'mu': mu_clamped})
import scipy.stats
# Computed the log joint manually
log_prior = scipy.stats.norm(mu_prior, sigma_prior).logpdf(mu_clamped)
log_lik  = np.sum(scipy.stats.norm(mu_clamped, sigma_x).logpdf(x))
log_joint = log_prior + log_lik
assert np.isclose(logp, log_joint)

 # Standard MCMC diagonistics
pm.traceplot(mcmc_samples)
pm.plot_posterior(mcmc_samples);
Rhat = pm.gelman_rubin(mcmc_samples)
print(Rhat)

# Estimate posterior over mu when unclamped
# Bayes rule for Gaussians MLAPA sec 5.6.2
Sigma_post = 1/( 1/Sigma_prior + N/Sigma_x )
xbar = np.mean(x)
mu_post = Sigma_post * (1/Sigma_x * N * xbar + 1/Sigma_prior * mu_prior)

vals = mcmc_samples.get_values('mu')
mu_post_mcmc = np.mean(vals)
Sigma_post_mcmc = np.var(vals)
assert np.isclose(mu_post, mu_post_mcmc, atol=1e-1)
assert np.isclose(Sigma_post, Sigma_post_mcmc, atol=1e-1)


pm.plot_posterior(vi_samples);


B = np.reshape(np.arange(6), (2,3))
a  = np.array([1,2]) # vector
A = np.array[a].T # (2,1) matrix
C = A * B # broadcast across columns
print(C)
      