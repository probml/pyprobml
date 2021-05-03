import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pyro
from pyro.distributions import Beta, Binomial
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import summary
from scipy import stats
 
import pyprobml_utils as pml

def partial_pooled(Y):
  with pyro.plate(Y.shape[0]):
    θ = Beta(1,1)
    return pyro.sample("y_obs", Binomial(total_count=1, probs=θ))

def get_summary_table(posterior, sites):
    """
    Return summarized statistics for each of the ``sites`` in the
    traces corresponding to the approximate posterior.
    """
    site_stats = {}
    for site_name in sites:
        marginal_site = posterior[site_name].cpu()

        site_summary = summary({site_name: marginal_site}, prob=0.5, group_by_chain=group_by_chain)[site_name]
        if site_summary["mean"].shape:
            site_df = pd.DataFrame(site_summary)
        else:
            site_df = pd.DataFrame(site_summary, index=[0])
        site_stats[site_name] = site_df.astype(float).round(2)
    return site_stats

def sample_posterior_pred(model, posterior_samples, y_obs):
  posterior_pred = Predictive(model=model, posterior_samples=posterior_samples)(y_obs, None)
  posterior_summary = get_summary_table(posterior_pred, sites=["y_obs"])
  return posterior_summary


def sample_prior_pred(model, numofsamples, y_obs):
  prior_pred = Predictive(model, {}, num_samples=numofsamples)
  prior_summary = get_summary_table(prior_pred, sites=["y_obs"])
  return prior_summary

np.random.seed(0)
Y = torch.Tensor(stats.bernoulli(0.7).rvs(20))

nuts_kernel = NUTS(partial_pooled)
mcmc = MCMC(nuts_kernel, 1000, num_chains=1)
mcmc.run(Y)
trace = mcmc.get_samples()
idata = az.from_pyro(trace)
observedY = partial_pooled(Y)
pred_dist = (sample_prior_pred(partial_pooled, 1000, observedY)  ,sample_posterior_pred(partial_pooled, idata, observedY))

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

fig, ax = plt.subplots()
az.plot_dist(θ.distribution.random(size=1000), plot_kwargs={"color":"0.5"}, fill_kwargs={'alpha':1})
ax.set_title("Prior distribution", fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 4)
ax.tick_params(axis='both', pad=7)
ax.set_xlabel("θ")

fig, ax = plt.subplots()
az.plot_dist(idata.posterior["θ"], plot_kwargs={"color":"0.5"}, fill_kwargs={'alpha':1})
ax.set_title("Posterior distribution", fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 4)
ax.tick_params(axis='both', pad=7)
ax.set_xlabel("θ")
