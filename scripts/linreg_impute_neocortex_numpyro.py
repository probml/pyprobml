# This file shows a demo implimentation of bayesian imputation for
# neocortex data
import superimport

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import random, ops

import arviz as az

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS
from linreg_impute_numpyro import linreg_imputation_model

az.style.use("arviz-darkgrid")
numpyro.set_host_device_count(4)

'''
# code taken from 
# https://fehiepsi.github.io/rethinking-numpyro/15-missing-data-and-other-opportunities.html#Code-15.22
'''
def rethinking_model(B, M, K):
    # priors
    a = numpyro.sample("a", dist.Normal(0, 0.5))
    muB = numpyro.sample("muB", dist.Normal(0, 0.5))
    muM = numpyro.sample("muM", dist.Normal(0, 0.5))
    bB = numpyro.sample("bB", dist.Normal(0, 0.5))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    Rho_BM = numpyro.sample("Rho_BM", dist.LKJ(2, 2))
    Sigma_BM = numpyro.sample("Sigma_BM", dist.Exponential(1).expand([2]))

    # define B_merge as mix of observed and imputed values
    B_impute = numpyro.sample(
        "B_impute", dist.Normal(0, 1).expand([int(np.isnan(B).sum())]).mask(False)
    )
    B_merge = ops.index_update(B, np.nonzero(np.isnan(B))[0], B_impute)

    # M and B correlation
    MB = jnp.stack([M, B_merge], axis=1)
    cov = jnp.outer(Sigma_BM, Sigma_BM) * Rho_BM
    numpyro.sample("MB", dist.MultivariateNormal(jnp.stack([muM, muB]), cov), obs=MB)

    # K as function of B and M
    mu = a + bB * B_merge + bM * M
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)


# To test whether our model is giving results to the rethinking_model

# fits the demo 2-D data and gives posterior samples.
def impute_and_fit_rethinking(X, y):
    mcmc = MCMC(NUTS(rethinking_model), num_warmup=500, num_samples=500, num_chains=4)
    mcmc.run(random.PRNGKey(0), X[:, 0], X[:, 1], y)  # model_params are passed
    return mcmc


def impute_and_fit(X, y):
    mcmc = MCMC(NUTS(linreg_imputation_model), num_warmup=500, num_samples=500, num_chains=4)
    mcmc.run(random.PRNGKey(0), X, y)
    return mcmc


# 2-D data from "rethinking-numpyro"
url = "https://raw.githubusercontent.com/fehiepsi/rethinking-numpyro/master/data/milk.csv"
df = pd.read_csv(url, sep=';')
df["neocortex.prop"] = df["neocortex.perc"] / 100
df["logmass"] = df.mass.apply(math.log)
k = df["kcal.per.g"].pipe(lambda x: (x - x.mean()) / x.std()).values
MB = np.zeros((k.shape[0], 2))
MB[:, 0] = df["neocortex.prop"].pipe(lambda x: (x - x.mean()) / x.std()).values
MB[:, 1] = df.logmass.pipe(lambda x: (x - x.mean()) / x.std()).values

m_rethinking = impute_and_fit_rethinking(MB, k)
m = impute_and_fit(MB, k)

# posteriors
post_rethinking = m_rethinking.get_samples()
post = m.get_samples()

print("beta[0]~bB, beta[1]~bM, rho~Rho-BM, sigma_y~sigma")
# box-plots of the corresponding posteriors of two models
az.plot_forest(
    [az.from_numpyro(m_rethinking), az.from_numpyro(m)],
    model_names=["rethinking model", "our imputation model"],
    var_names=["a","beta", "bB", "bM", "sigma_y", "sigma", "rho", "Rho_BM"],
    combined=True,
    hdi_prob=0.90,
)
plt.show()

# testing whether the posterior samples are similar
# by comparing their diagnostics

a_rethinking = np.mean(post_rethinking['a'])
betaB_rethinking_samples = post_rethinking['bB']
betaB_rethinking = np.mean(betaB_rethinking_samples)
betaM_rethinking_samples = post_rethinking['bM']
betaM_rethinking = np.mean(betaM_rethinking_samples)
sigma_y_rethinking = np.mean(post_rethinking['sigma'])
rho_rethinking = np.mean(post_rethinking['Rho_BM'],axis=0)

a = np.mean(post['a'])
beta_samples = post['beta']
betaB = np.mean(beta_samples[:,0])
betaM = np.mean(beta_samples[:,1])
sigma_y = np.mean(post['sigma_y'])
rho = np.mean(post['rho'],axis=0)

print("a: ",[a,a_rethinking])
assert np.isclose(a, a_rethinking, atol=1e-1)
print("beta-B: ",[betaB, betaB_rethinking])
assert np.isclose(betaB, betaB_rethinking, atol=1e-1)
print("beta_M: ",[betaM, betaM_rethinking])
assert np.isclose(betaM, betaM_rethinking, atol=1e-1)
print("sigma_y: ",[sigma_y, sigma_y_rethinking])
assert np.isclose(sigma_y, sigma_y_rethinking, atol=1e-1)
for i in range(2):
  for j in range(2):
    print("rho[{},{}]: ".format(i,j),[rho[i][j], rho_rethinking[i][j]])
    assert np.isclose(rho[i][j], rho_rethinking[i][j], atol=1e-1)
