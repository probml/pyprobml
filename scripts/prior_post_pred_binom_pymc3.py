# prior and posterior predctiive for beta binomial
# fig 1.6 of 'Bayeysian Modeling and Computation'

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

#plt.figure()
#plt.plot(Y)
#pml.savefig("Bayesian_quartet_distributions.pdf", dpi=300)


with pm.Model() as model:
    θ = pm.Beta("θ", 1, 1)
    y_obs = pm.Binomial("y_obs",n=1, p=θ, observed=Y)
    idata = pm.sample(1000, return_inferencedata=True);

'''
pred_dists = (pm.sample_prior_predictive(1000, model)["y_obs"],
              pm.sample_posterior_predictive(idata, 1000, model)["y_obs"])


fig, ax = plt.subplots(4, 1, figsize=(9, 9))

for idx, n_d, dist in zip((1, 3), ("Prior", "Posterior"), pred_dists):
    az.plot_dist(dist.sum(1), hist_kwargs={"color":"0.5", "bins":range(0, 22)},
                                           ax=ax[idx])
    ax[idx].set_title(f"{n_d} predictive distribution",fontweight='bold')
    ax[idx].set_xlim(-1, 21)
    ax[idx].set_ylim(0, 0.15) 
    ax[idx].set_xlabel("number of success")

az.plot_dist(θ.distribution.random(size=1000), plot_kwargs={"color":"0.5"},
             fill_kwargs={'alpha':1}, ax=ax[0])
ax[0].set_title("Prior distribution", fontweight='bold')
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 4)
ax[0].tick_params(axis='both', pad=7)
ax[0].set_xlabel("θ") 
    
az.plot_dist(idata.posterior["θ"], plot_kwargs={"color":"0.5"},
             fill_kwargs={'alpha':1}, ax=ax[2])
ax[2].set_title("Posterior distribution", fontweight='bold')
ax[2].set_xlim(0, 1)
ax[2].set_ylim(0, 4)
ax[2].tick_params(axis='both', pad=7)
ax[2].set_xlabel("θ")


#pml.savefig("prior_post_pred_binom.pdf")
plt.savefig("prior_post_pred_binom.pdf")

plt.show()
'''