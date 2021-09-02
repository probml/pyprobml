# Illustrate Neal's funnel
# Code based on
#https://numpyro.readthedocs.io/en/latest/examples/funnel.html

import superimport

import pyprobml_utils as pml

import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam


def model():
    y = numpyro.sample("y", dist.Normal(0, 3))
    numpyro.sample("x", dist.Normal(0, jnp.exp(y/2)))

def noncentered_model():
    y = numpyro.sample("y", dist.Normal(0, 3))
    noise = numpyro.sample("noise", dist.Normal(0,1))
    x = numpyro.deterministic("x", noise*jnp.exp(y/2))

reparam_model = reparam(model, config={"x": LocScaleReparam(0)})


def run_inference(model):
    kernel = NUTS(model)
    rng_key = random.PRNGKey(0)
    mcmc = MCMC(kernel, num_warmup = 500, num_samples = 500, num_chains = 1)
    mcmc.run(rng_key)
    mcmc.print_summary(exclude_deterministic=False)
    return mcmc.get_samples()


print('centered')
samples = run_inference(model)

print('non-centered')
noncentered_samples = run_inference(noncentered_model)
noncentered_samples = Predictive(
    noncentered_model, noncentered_samples, return_sites=["x", "y"]
)(random.PRNGKey(1))

print('reparam')
reparam_samples = run_inference(reparam_model)
reparam_samples = Predictive(
    reparam_model, reparam_samples, return_sites=["x", "y"]
)(random.PRNGKey(1))

fig, ax = plt.subplots()
ax.plot(samples["x"], samples["y"], "go", alpha=0.3)
ax.set(
    xlim=(-20, 20),
    ylim=(-9, 9),
    title="Funnel samples with centered parameterization",
)
pml.savefig("funnel_plot_centered.pdf")

fig, ax = plt.subplots()
ax.plot(noncentered_samples["x"], noncentered_samples["y"], "go", alpha=0.3)
ax.set(
    xlim=(-20, 20),
    ylim=(-9, 9),
    title="Funnel samples with non-centered parameterization",
)
pml.savefig("funnel_plot_noncentered.pdf")

fig, ax = plt.subplots()
ax.plot(reparam_samples["x"], reparam_samples["y"], "go", alpha=0.3)
ax.set(
    xlim=(-20, 20),
    ylim=(-9, 9),
    title="Funnel samples with reparameterization",
)
pml.savefig("funnel_plot_reparam.pdf")