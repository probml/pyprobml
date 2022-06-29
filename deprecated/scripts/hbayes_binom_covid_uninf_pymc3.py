import superimport

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
#import seaborn as sns
import pymc3 as pm
import arviz as az
import theano.tensor as tt
import pyprobml_utils as pml

np.random.seed(123)

G_samples = np.array([0, 0, 2, 0, 1, 1, 0, 2, 1, 3, 0, 1, 1, 1,
                      54, 0, 0, 1, 3, 0])
N_samples = np.array([1083, 855, 3461, 657, 1208, 1025, 527,
                      1668, 583, 582, 917, 857,
    680, 917, 53637, 874, 395, 581, 588, 383])

G_samples = np.array([1,    0,    3,    0,   1,    5,     11])
N_samples = np.array([1083, 855, 3461, 657, 1208, 5000, 10000])

if False:
    with pm.Model() as model:
        μ = pm.Beta('μ', 1., 1.)
        κ = pm.HalfNormal('κ', 10)
        theta = pm.Beta('θ', alpha=μ*κ, beta=(1.0-μ)*κ, shape=len(N_samples))
        #y = pm.Bernoulli('y', p=θ[group_idx], observed=data)
        y = pm.Binomial('y', p=theta, observed=G_samples, n=N_samples)
        trace = pm.sample(1000, cores=1, chains=2)
    
#https://docs.pymc.io/notebooks/GLM-hierarchical-binominal-model.html
    

def logp_ab(value):
    ''' prior density'''
    return tt.log(tt.pow(tt.sum(value), -5/2))

with pm.Model() as model:
    # Uninformative prior for alpha and beta
    ab = pm.HalfFlat('ab',
                     shape=2,
                     testval=np.asarray([1., 1.]))
    pm.Potential('p(a, b)', logp_ab(ab))
    alpha = pm.Deterministic('alpha', ab[0])
    beta = pm.Deterministic('beta', ab[1])
    theta = pm.Beta('θ', alpha=alpha, beta=beta, shape=len(N_samples))
    y = pm.Binomial('y', p=theta, observed=G_samples, n=N_samples)
    trace = pm.sample(1000, cores=1, chains=2)
    
az.plot_trace(trace)
pml.savefig('hbayes_binom_covid_trace.pdf', dpi=300)

print(az.summary(trace))

axes = az.plot_forest(
    trace, var_names='θ', hdi_prob=0.95, combined=False, colors='cycle')
y_lims = axes[0].get_ylim()
pml.savefig('hbayes_binom_covid_forest.pdf')

J = len(N_samples)
post_mean = np.zeros(J)
samples = trace['θ']
post_mean = np.mean(samples, axis=0)

alphas = trace['alpha']
betas = trace['beta']
alpha_mean = np.mean(alphas)
beta_mean = np.mean(betas)
hyper_mean = alpha_mean/(alpha_mean + beta_mean)
print('hyper mean')
print(hyper_mean)
hyper_mean2 = np.mean(alphas / (alphas+betas))
print(hyper_mean2)

mle = G_samples / N_samples
pooled_mle = np.sum(G_samples) / np.sum(N_samples)

print('pooled mle')
print(pooled_mle)

fig, axs = plt.subplots(4,1, figsize=(8,8))
axs = np.reshape(axs, 4)
xs = np.arange(J)
ax = axs[0]
ax.bar(xs, G_samples)
ax.set_ylim(0, 5)
ax.set_title('number of cases (truncated at 5)')
ax = axs[1]
ax.bar(xs, N_samples)
ax.set_ylim(0, 1000)
ax.set_title('popn size (truncated at 1000)')
ax = axs[2]
ax.bar(xs, mle)
ax.hlines(pooled_mle, 0, J, 'r', lw=3)
ax.set_title('MLE (red line = pooled)')
ax = axs[3]
ax.bar(xs, post_mean)
ax.hlines(hyper_mean, 0, J, 'r', lw=3)
ax.set_title('posterior mean (red line = hparam)')
pml.savefig('hbayes_binom_covid_barplot.pdf')

plt.show()