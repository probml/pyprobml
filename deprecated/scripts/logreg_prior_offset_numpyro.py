
# We illustrate effect of prior on parameters for logistic regression
# Based on fig 11.3 of
# [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/).
# The numpyro code is nased on [Du Phan's site]
# https://fehiepsi.github.io/rethinking-numpyro/11-god-spiked-the-integers.html

import superimport

import pyprobml_utils as pml

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
from numpyro.infer import Predictive

import arviz as az

from jax.scipy.special import expit
from functools import partial

### Model with just offset term

def model_meta(prior_std, obs=None):
    a = numpyro.sample("a", dist.Normal(0, prior_std))
    numpyro.sample("obs", dist.Binomial(logits=a), obs=obs)

fig, ax = plt.subplots()
colors = ['r', 'k']
for i, sigma in enumerate([1.5, 10]):
  model = partial(model_meta, sigma)
  prior = Predictive(model, num_samples=10000)(random.PRNGKey(1999))
  p = expit(prior["a"])
  label = r'variance={:0.2f}$'.format(sigma)
  az.plot_kde(p, ax=ax, plot_kwargs = {'color': colors[i]}, label=label, legend=True)
pml.savefig('logreg_prior_offset.pdf', dpi=300)
plt.show()
