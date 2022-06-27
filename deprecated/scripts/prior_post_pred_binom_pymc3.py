# prior and posterior predctiive for beta binomial
# fig 1.6 of 'Bayeysian Modeling and Computation'

import superimport

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats
from scipy.stats import entropy
from scipy.optimize import minimize
import pyprobml_utils as pml


np.random.seed(0)
Y = stats.bernoulli(0.7).rvs(20)



with pm.Model() as model:
    θ = pm.Beta("θ", 1, 1)
    y_obs = pm.Binomial("y_obs",n=1, p=θ, observed=Y)
    trace = pm.sample(1000, cores=1, chains=2, return_inferencedata=False)

idata = az.from_pymc3(trace)

pred_dists = (pm.sample_prior_predictive(1000, model)["y_obs"],
              pm.sample_posterior_predictive(idata, 1000, model)["y_obs"])

dist=pred_dists[0]
print(dist.shape)
num_success = dist.sum(1)
print(num_success.shape)

fig, ax = plt.subplots()
az.plot_dist(pred_dists[0].sum(1), hist_kwargs={"color":"0.5", "bins":range(0, 22)})
ax.set_title(f"Prior predictive distribution",fontweight='bold')
ax.set_xlim(-1, 21)
ax.set_ylim(0, 0.15) 
ax.set_xlabel("number of success")

fig, ax = plt.subplots()
az.plot_dist(pred_dists[1].sum(1), hist_kwargs={"color":"0.5", "bins":range(0, 22)})
ax.set_title(f"Posterior predictive distribution",fontweight='bold')
ax.set_xlim(-1, 21)
ax.set_ylim(0, 0.15) 
ax.set_xlabel("number of success")
pml.savefig('Posterior_predictive_distribution.pdf')


fig, ax = plt.subplots()
az.plot_dist(θ.distribution.random(size=1000), plot_kwargs={"color":"0.5"},
             fill_kwargs={'alpha':1})
ax.set_title("Prior distribution", fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 4)
ax.tick_params(axis='both', pad=7)
ax.set_xlabel("θ")
pml.savefig('Prior_distribution.pdf')
fig, ax = plt.subplots()
az.plot_dist(idata.posterior["θ"], plot_kwargs={"color":"0.5"},
             fill_kwargs={'alpha':1})
ax.set_title("Posterior distribution", fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 4)
ax.tick_params(axis='both', pad=7)
ax.set_xlabel("θ")
pml.savefig('Posterior_distribution.pdf')

plt.show()