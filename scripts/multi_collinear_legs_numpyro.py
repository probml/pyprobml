

#We illustrate multicollinearity using the example in sec 6.1  of
# [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/).
#The numpyro code is from [Du Phan's site]
# (https://fehiepsi.github.io/rethinking-numpyro/06-the-haunted-dag-and-the-causal-terror.html)

import superimport
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math
import os
import warnings
import pandas as pd

import jax
print("jax version {}".format(jax.__version__))
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))

import jax.numpy as jnp
from jax import random, vmap

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform
from numpyro.diagnostics import hpdi, print_summary
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
import numpyro.optim as optim


import arviz as az

import pyprobml_utils as pml

# Data

def sample_data1():
    N = 100  # number of individuals
    with numpyro.handlers.seed(rng_seed=909):
        # sim total height of each
        height = numpyro.sample("height", dist.Normal(10, 2).expand([N]))
        # leg as proportion of height
        leg_prop = numpyro.sample("prop", dist.Uniform(0.4, 0.5).expand([N]))
        # sim right leg as proportion + error
        leg_right = leg_prop * height + numpyro.sample(
            "right_error", dist.Normal(0, 0.02).expand([N])
        )
        # sim left leg as proportion + error
        leg_left = leg_prop * height + numpyro.sample(
            "left_error", dist.Normal(0, 0.02).expand([N])
        )
        # combine into data frame
        d = pd.DataFrame({"height": height, "leg_left": leg_left, "leg_right": leg_right})

        return d

def sample_data2():
    N = 100  # number of individuals
    # sim total height of each
    height = dist.Normal(10, 2).sample(random.PRNGKey(0), (N,))
    # leg as proportion of height
    leg_prop = dist.Uniform(0.4, 0.5).sample(random.PRNGKey(1), (N,))
    # sim left leg as proportion + error
    leg_left = leg_prop * height + dist.Normal(0, 0.02).sample(random.PRNGKey(2), (N,))
    # sim right leg as proportion + error
    leg_right = leg_prop * height + dist.Normal(0, 0.02).sample(random.PRNGKey(3), (N,))
    # combine into data frame
    d = pd.DataFrame({"height": height, "leg_left": leg_left, "leg_right": leg_right})
    return d

df = sample_data2()

# Model

def model_book(leg_left, leg_right, height, br_positive=False):
    a = numpyro.sample("a", dist.Normal(10, 100))
    bl = numpyro.sample("bl", dist.Normal(2, 10))
    if br_positive:
        br = numpyro.sample("br", dist.TruncatedNormal(0, 2, 10))
    else:
        br = numpyro.sample("br", dist.Normal(2, 10))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bl * leg_left + br * leg_right
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)

def model_vague_prior(leg_left, leg_right, height, br_positive=False):
    # we modify the priors to make them less informative
    a = numpyro.sample("a", dist.Normal(0, 100))
    bl = numpyro.sample("bl", dist.Normal(0, 100))
    if br_positive:
        br = numpyro.sample("br", dist.TruncatedNormal(0, 0, 100))
    else:
        br = numpyro.sample("br", dist.Normal(0, 100))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bl * leg_left + br * leg_right
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)

model = model_vague_prior

# Analyse posterior

def analyze_post(post, method):
    print_summary(post, 0.95, False)
    fig, ax = plt.subplots()
    az.plot_forest(post, hdi_prob=0.95, figsize=(10, 4), ax=ax)
    plt.title(method)
    pml.savefig(f'multicollinear_forest_plot_{method}.pdf')
    plt.show()

    # post = m6_1.sample_posterior(random.PRNGKey(1), p6_1, (1000,))
    fig, ax = plt.subplots()
    az.plot_pair(post, var_names=["br", "bl"],
                 scatter_kwargs={"alpha": 0.1}, ax=ax)
    pml.savefig(f'multicollinear_joint_post_{method}.pdf')
    plt.title(method)
    plt.show()

    sum_blbr = post["bl"] + post["br"]
    fig, ax = plt.subplots()
    az.plot_kde(sum_blbr, label="sum of bl and br", ax=ax)
    plt.title(method)
    pml.savefig(f'multicollinear_sum_post_{method}.pdf')
    plt.show()

# Laplace fit

m6_1 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m6_1,
    optim.Adam(0.1),
    Trace_ELBO(),
    leg_left=df.leg_left.values,
    leg_right=df.leg_right.values,
    height=df.height.values,
    br_positive=False
)
svi_run = svi.run(random.PRNGKey(0), 2000)
p6_1 = svi_run.params
losses = svi_run.losses
post_laplace = m6_1.sample_posterior(random.PRNGKey(1), p6_1, (1000,))

analyze_post(post_laplace, 'laplace')


# MCMC fit
# code from p298 (code 9.28) of rethinking2
#https://fehiepsi.github.io/rethinking-numpyro/09-markov-chain-monte-carlo.html


kernel = NUTS(
    model,
    init_strategy=init_to_value(values={"a": 10.0, "bl": 0.0, "br": 0.1, "sigma": 1.0}),
)
mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=4)
# df.T has size 3x100
data_dict = dict(zip(df.columns, df.T.values))
data_dict['br_positive'] = False
mcmc.run(random.PRNGKey(0), **data_dict)

mcmc.print_summary()
post_hmc  = mcmc.get_samples()
analyze_post(post_hmc, 'hmc')

# Constrained model where beta_r >= 0

data_dict = dict(zip(df.columns, df.T.values))
data_dict['br_positive'] = True
mcmc.run(random.PRNGKey(0), **data_dict)

mcmc.print_summary()
post_hmc  = mcmc.get_samples()
analyze_post(post_hmc, 'hmc_br_pos')