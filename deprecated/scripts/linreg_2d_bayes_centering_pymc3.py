# Illustrate benefits of centering data
# for reducing correlation between slope and intercept for 1d regression

# Based on 
#https://github.com/aloctavodia/BAP/blob/master/code/Chp3/03_Modeling%20with%20Linear%20Regressions.ipynb

import superimport

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pyprobml_utils as pml
import os

    
np.random.seed(1)
N = 100
alpha_real = 2.5
beta_real = 0.9
noiseSD = 0.5
eps_real = np.random.normal(0, noiseSD, size=N)

x = np.random.normal(10, 1, N) # centered on 10
y_real = alpha_real + beta_real * x
y = y_real + eps_real

# save untransformed data for later
x_orig = x
y_orig = y

_, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(x, y, 'C0.')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation=0)
ax[0].plot(x, y_real, 'k')
az.plot_kde(y, ax=ax[1])
ax[1].set_xlabel('y')
plt.tight_layout()


# Fit posterior with MCMC instead of analytically (for simplicity and flexibility)
# This is the same as BAP code, except we fix the noise variance to a constant.

with pm.Model() as model_g:
    w0 = pm.Normal('w0', mu=0, sd=10)
    w1 = pm.Normal('w1', mu=0, sd=1)
    #ϵ = pm.HalfCauchy('ϵ', 5)
    mu = pm.Deterministic('mu', w0 + w1 * x)
    #y_pred = pm.Normal('y_pred', mu=μ, sd=ϵ, observed=y)
    y_pred = pm.Normal('y_pred', mu=mu, sd=noiseSD, observed=y)
    trace_g = pm.sample(1000, cores=1, chains=2)

az.plot_trace(trace_g, var_names=['w0', 'w1'])

az.plot_pair(trace_g,var_names=['w0', 'w1'], plot_kwargs={'alpha': 0.1});
pml.savefig('linreg_2d_bayes_post_noncentered_data.pdf')
plt.show()


# To reduce the correlation between alpha and beta, we can center the data
x = x_orig - x_orig.mean()

# or standardize the data
#x = (x - x.mean())/x.std()
#y = (y - y.mean())/y.std()

with pm.Model() as model_g_centered:
    w0 = pm.Normal('w0', mu=0, sd=10)
    w1 = pm.Normal('w1', mu=0, sd=1)
    #ϵ = pm.HalfCauchy('ϵ', 5)
    mu = pm.Deterministic('mu', w0 + w1 * x)
    #y_pred = pm.Normal('y_pred', mu=μ, sd=ϵ, observed=y)
    y_pred = pm.Normal('y_pred', mu=mu, sd=noiseSD, observed=y)
    trace_g_centered = pm.sample(1000, cores=1, chains=2)


az.plot_pair(trace_g_centered, var_names=['w0', 'w1'], plot_kwargs={'alpha': 0.1});
pml.savefig('linreg_2d_bayes_post_centered_data.pdf')
plt.show()

