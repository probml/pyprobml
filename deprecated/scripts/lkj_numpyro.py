
# We illustrate LKJ prior as discussed in fig 14.3 of
# [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/).
# The numpyro code is from [Du Phan's site]
#https://fehiepsi.github.io/rethinking-numpyro/14-adventures-in-covariance.html

import superimport

import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt

import jax
print("jax version {}".format(jax.__version__))
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))

import jax.numpy as jnp
from jax import random, vmap

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

import numpyro
import numpyro.distributions as dist

import arviz as az

import pyprobml_utils as pml

eta_list = [1,2,4]
colors = ['r', 'k', 'b']
fig, ax = plt.subplots()
for i, eta in enumerate(eta_list):
  R = dist.LKJ(dimension=2, concentration=eta).sample(random.PRNGKey(0), (int(1e4),))
  az.plot_kde(R[:, 0, 1], label=f"eta={eta}", plot_kwargs  ={'color': colors[i]})
plt.legend()
ax.set_xlabel('correlation')
ax.set_ylabel('density')
ax.set_ylim(0, 1.2)
ax.set_xlim(-1.1, 1.1)
pml.savefig('LKJ_1d_correlation.pdf', dpi=300)
plt.show()

