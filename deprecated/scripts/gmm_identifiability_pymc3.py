# Gaussian mixture model suign PyMC3
# Based on https://github.com/aloctavodia/BAP/blob/master/code/Chp6/06_mixture_models.ipynb

import superimport

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import pandas as pd
import theano.tensor as tt
import matplotlib.pyplot as plt
import arviz as az
import pyprobml_utils as pml

np.random.seed(42)


#url = 'https://github.com/aloctavodia/BAP/tree/master/code/data/chemical_shifts_theo_exp.csv?raw=true'
# There is some error reading the abvoe file
# Error tokenizing data. C error: Expected 1 fields in line 71, saw 2
# So we make a copy here
url = 'https://github.com/probml/pyprobml/blob/master/data/chemical_shifts_theo_exp.csv?raw=true'
df = pd.read_csv(url)
obs = df['exp']

az.plot_kde(obs)
plt.hist(obs, density=True, bins=30, alpha=0.3)
plt.yticks([])
pml.savefig('gmm_pymc3_data.pdf', dpi=300)

# Illustrate unidentifiability

clusters = 2
with pm.Model() as model_mg:
    p = pm.Dirichlet('p', a=np.ones(clusters))
    means = pm.Normal('means', mu=obs.mean(), sd=10, shape=clusters)
    sd = pm.HalfNormal('sd', sd=10)
    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=obs)
    trace_mg = pm.sample(random_seed=123, cores=1, chains=2)

varnames = ['means', 'p']
print(az.summary(trace_mg, varnames))

az.plot_trace(trace_mg, varnames)
pml.savefig('gmm_pymc3_label_switching.pdf', dpi=300)

# Add constraint that mu[0] < mu[1] using a potential (penalty) function

clusters = 2
with pm.Model() as model_mgp:
    p = pm.Dirichlet('p', a=np.ones(clusters))
    means = pm.Normal('means', mu=np.array([.9, 1]) * obs.mean(),
                      sd=10, shape=clusters)
    sd = pm.HalfNormal('sd', sd=10)
    order_means = pm.Potential('order_means',
                               tt.switch(means[1]-means[0] < 0,
                                         -np.inf, 0))
    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=obs)
    trace_mgp = pm.sample(1000, random_seed=123, cores=1, chains=2)

varnames = ['means', 'p']
print(az.summary(trace_mgp, varnames))
az.plot_trace(trace_mgp, varnames)
pml.savefig('gmm_pymc3_constrained.pdf', dpi=300)

plt.show()