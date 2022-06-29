# Hierarchcal Bayesian linear regression on 1d synthettic data
# Based on: https://github.com/aloctavodia/BAP/blob/master/code/Chp3/03_Modeling%20with%20Linear%20Regressions.ipynb


import superimport

import numpy as np
import matplotlib.pyplot as plt

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pyprobml_utils as pml
    
N = 10 # nun samples per group
M = 8 # num groups
idx = np.repeat(range(M-1), N) # N samples for groups 0-6
idx = np.append(idx, 7) # 1 sample for group 7
np.random.seed(314)

alpha_real = np.random.normal(2.5, 0.5, size=M)
beta_real = np.random.beta(6, 1, size=M)
eps_real = np.random.normal(0, 0.5, size=len(idx))

y_m = np.zeros(len(idx))
x_m = np.random.normal(10, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real

_, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
ax = np.ravel(ax)
j, k = 0, N
for i in range(M):
    ax[i].scatter(x_m[j:k], y_m[j:k])
    ax[i].set_xlabel(f'x_{i}')
    ax[i].set_ylabel(f'y_{i}', rotation=0, labelpad=15)
    ax[i].set_xlim(6, 15)
    ax[i].set_ylim(7, 17)
    j += N
    k += N
plt.tight_layout()
pml.savefig('linreg_hbayes_1d_data.pdf')



x_centered = x_m - x_m.mean()


with pm.Model() as unpooled_model:
    α_tmp = pm.Normal('α_tmp', mu=0, sd=10, shape=M)
    β = pm.Normal('β', mu=0, sd=10, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)
    ν = pm.Exponential('ν', 1/30)

    y_pred = pm.StudentT('y_pred', mu=α_tmp[idx] + β[idx] * x_centered,
                         sd=ϵ, nu=ν, observed=y_m)

    α = pm.Deterministic('α', α_tmp - β * x_m.mean())

    trace_up = pm.sample(2000, cores=1, chains=2)


az.summary(trace_up)


def plot_regression_line(trace):
    _, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True,
                       constrained_layout=True)
    ax = np.ravel(ax)
    j, k = 0, N
    x_range = np.linspace(x_m.min(), x_m.max(), 10)
    for i in range(M):
        ax[i].scatter(x_m[j:k], y_m[j:k])
        ax[i].set_xlabel(f'x_{i}')
        ax[i].set_ylabel(f'y_{i}', labelpad=17, rotation=0)
        alpha_m = trace['α'][:, i].mean()
        beta_m = trace['β'][:, i].mean()
        ax[i].plot(x_range, alpha_m + beta_m * x_range, c='k',
                  label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
        plt.xlim(x_m.min()-1, x_m.max()+1)
        plt.ylim(y_m.min()-1, y_m.max()+1)
        ax[i].legend()
        j += N
        k += N
      


plot_regression_line(trace_up)    
pml.savefig('linreg_hbayes_1d_unpooled_mean.pdf')

def plot_post_pred_samples(trace, nsamples=20):
    _, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True,
                       constrained_layout=True)
    ax = np.ravel(ax)
    j, k = 0, N
    x_range = np.linspace(x_m.min(), x_m.max(), 10)
    X =  x_range[:, np.newaxis]
    
    for i in range(M):
        ax[i].scatter(x_m[j:k], y_m[j:k])
        ax[i].set_xlabel(f'x_{i}')
        ax[i].set_ylabel(f'y_{i}', labelpad=17, rotation=0)
        alpha_m = trace['α'][:, i].mean()
        beta_m = trace['β'][:, i].mean()
        ax[i].plot(x_range, alpha_m + beta_m * x_range, c='r', lw=3,
                  label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
        plt.xlim(x_m.min()-1, x_m.max()+1)
        plt.ylim(y_m.min()-1, y_m.max()+1)
        alpha_samples = trace['α'][:,i]
        beta_samples = trace['β'][:,i]
        ndx = np.random.choice(np.arange(len(alpha_samples)), nsamples)
        alpha_samples_thinned = alpha_samples[ndx]
        beta_samples_thinned = beta_samples[ndx]
        ax[i].plot(x_range, alpha_samples_thinned + beta_samples_thinned * X,
            c='gray', alpha=0.5)
        
        j += N
        k += N
    
plot_post_pred_samples(trace_up)    
pml.savefig('linreg_hbayes_1d_unpooled_samples.pdf')
    
    
with pm.Model() as hierarchical_model:
    # hyper-priors
    α_μ_tmp = pm.Normal('α_μ_tmp', mu=0, sd=10)
    α_σ_tmp = pm.HalfNormal('α_σ_tmp', 10)
    β_μ = pm.Normal('β_μ', mu=0, sd=10)
    β_σ = pm.HalfNormal('β_σ', sd=10)

    # priors
    α_tmp = pm.Normal('α_tmp', mu=α_μ_tmp, sd=α_σ_tmp, shape=M)
    β = pm.Normal('β', mu=β_μ, sd=β_σ, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)
    ν = pm.Exponential('ν', 1/30)

    y_pred = pm.StudentT('y_pred', mu=α_tmp[idx] + β[idx] * x_centered,
                         sd=ϵ, nu=ν, observed=y_m)

    # convert estimates to equivalent on uncentered data
    α = pm.Deterministic('α', α_tmp - β * x_m.mean()) 
    α_μ = pm.Deterministic('α_μ', α_μ_tmp - β_μ * x_m.mean())
    α_σ = pm.Deterministic('α_sd', α_σ_tmp - β_μ * x_m.mean())

    trace_hm = pm.sample(1000, cores=1, chains=2)

az.summary(trace_hm)



plot_regression_line(trace_hm)    
pml.savefig('linreg_hbayes_1d_pooled_mean.pdf')

plot_post_pred_samples(trace_hm)    
pml.savefig('linreg_hbayes_1d_pooled_samples.pdf')

plt.show()