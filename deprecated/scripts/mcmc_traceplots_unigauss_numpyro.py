
# We illustrate bad mixing MCMC chains using the example in sec 9.5  of
# [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/).
# The numpyro code is from [Du Phan's site]
# https://fehiepsi.github.io/rethinking-numpyro/09-markov-chain-monte-carlo.html


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

y = jnp.array([-1, 1])

# Model with vague priors

def model(y):
    alpha = numpyro.sample("alpha", dist.Normal(0, 1000))
    sigma = numpyro.sample("sigma", dist.Exponential(0.0001))
    mu = alpha
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


m9_2 = MCMC(
    NUTS(model, target_accept_prob=0.95), num_warmup=500, num_samples=500, num_chains=3
)
m9_2.run(random.PRNGKey(11), y=y)

m9_2.print_summary(0.95)

az.plot_trace(az.from_numpyro(m9_2))
pml.savefig('mcmc_traceplot_unigauss_bad.png')
plt.show()

az.plot_rank(az.from_numpyro(m9_2))
pml.savefig('mcmc_trankplot_unigauss_bad.png')
plt.show()

# Model with proper priors

def model(y):
    alpha = numpyro.sample("alpha", dist.Normal(1, 10))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = alpha
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


m9_3 = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=3)
m9_3.run(random.PRNGKey(11), y=y)
m9_3.print_summary(0.95)

az.plot_trace(az.from_numpyro(m9_3))
pml.savefig('mcmc_traceplot_unigauss_good.png')
plt.show()

az.plot_rank(az.from_numpyro(m9_3))
pml.savefig('mcmc_trankplot_unigauss_good.png')
plt.show()


