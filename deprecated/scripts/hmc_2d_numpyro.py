
# We illustrate HMC using the 2d example in sec 9.3  of
# [Statistical Rethinking ed 2](https://xcelab.net/rm/statistical-rethinking/).
# The numpyro code is from [Du Phan's site]
# (https://fehiepsi.github.io/rethinking-numpyro/09-markov-chain-monte-carlo.html)
                          
import superimport

import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math
import os
import warnings
import pandas as pd

import jax
from jax import lax, ops, random
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
from numpyro.infer import Predictive, log_likelihood
from numpyro.infer import MCMC, NUTS
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
import numpyro.optim as optim


import arviz as az

# Model

# U needs to return neg-log-probability
def U(q, a=0, b=1, k=0, d=1):
    muy = q[0]
    mux = q[1]
    logprob_y = jnp.sum(dist.Normal(muy, 1).log_prob(y))
    logprob_x = jnp.sum(dist.Normal(mux, 1).log_prob(x))
    logprob_muy = dist.Normal(a, b).log_prob(muy)
    logprob_mux = dist.Normal(k, d).log_prob(mux)
    U = logprob_y + logprob_x + logprob_muy + logprob_mux
    return -U

# gradient function
# need vector of partial derivatives of U with respect to vector q
def U_gradient(q, a=0, b=1, k=0, d=1):
    muy = q[0]
    mux = q[1]
    G1 = jnp.sum(y - muy) + (a - muy) / b ** 2  # dU/dmuy
    G2 = jnp.sum(x - mux) + (k - mux) / b ** 2  # dU/dmux
    return jnp.stack([-G1, -G2])  # negative bc energy is neg-log-prob


# test data
with numpyro.handlers.seed(rng_seed=7):
    y = numpyro.sample("y", dist.Normal().expand([50]))
    x = numpyro.sample("x", dist.Normal().expand([50]))
    x = (x - jnp.mean(x)) / jnp.std(x)
    y = (y - jnp.mean(y)) / jnp.std(y)
    
# Algorithm
    
def HMC2(U, grad_U, epsilon, L, current_q, rng):
    q = current_q
    # random flick - p is momentum
    p = dist.Normal(0, 1).sample(random.fold_in(rng, 0), (q.shape[0],))
    current_p = p
    # Make a half step for momentum at the beginning
    p = p - epsilon * grad_U(q) / 2
    # initialize bookkeeping - saves trajectory
    qtraj = jnp.full((L + 1, q.shape[0]), jnp.nan)
    ptraj = qtraj
    qtraj = ops.index_update(qtraj, 0, current_q)
    ptraj = ops.index_update(ptraj, 0, p)

    # Alternate full steps for position and momentum
    for i in range(L):
        q = q + epsilon * p  # Full step for the position
        # Make a full step for the momentum, except at end of trajectory
        if i != (L - 1):
            p = p - epsilon * grad_U(q)
            ptraj = ops.index_update(ptraj, i + 1, p)
        qtraj = ops.index_update(qtraj, i + 1, q)

    # Make a half step for momentum at the end
    p = p - epsilon * grad_U(q) / 2
    ptraj = ops.index_update(ptraj, L, p)
    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q)
    current_K = jnp.sum(current_p ** 2) / 2
    proposed_U = U(q)
    proposed_K = jnp.sum(p ** 2) / 2
    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    accept = 0
    runif = dist.Uniform().sample(random.fold_in(rng, 1))
    if runif < jnp.exp(current_U - proposed_U + current_K - proposed_K):
        new_q = q  # accept
        accept = 1
    else:
        new_q = current_q  # reject
    return {
        "q": new_q,
        "traj": qtraj,
        "ptraj": ptraj,
        "accept": accept,
        "dH": proposed_U + proposed_K - (current_U + current_K),
    }

def make_plot(step=0.03, L=11):
  Q = {}
  Q["q"] = jnp.array([-0.1, 0.2])
  pr = 0.4 #0.31
  plt.figure()
  plt.subplot(ylabel=r"$\mu_y$", xlabel=r"$\mu_x$", xlim=(-pr, pr), ylim=(-pr, pr))
  n_samples = 4
  path_col = (0, 0, 0, 0.5)
  for r in 0.075 * jnp.arange(2, 6):
      plt.gca().add_artist(plt.Circle((0, 0), r, alpha=0.2, fill=False))
  plt.scatter(Q["q"][0], Q["q"][1], c="k", marker="x", zorder=4)
  for i in range(n_samples):
      Q = HMC2(U, U_gradient, step, L, Q["q"], random.fold_in(random.PRNGKey(0), i))
      if n_samples < 10:
          for j in range(L):
              K0 = jnp.sum(Q["ptraj"][j] ** 2) / 2
              plt.plot(
                  Q["traj"][j : j + 2, 0],
                  Q["traj"][j : j + 2, 1],
                  c=path_col,
                  lw=1 + 2 * K0,
              )
          plt.scatter(Q["traj"][:, 0], Q["traj"][:, 1], c="white", s=5, zorder=3)
          # for fancy arrows
          dx = Q["traj"][L, 0] - Q["traj"][L - 1, 0]
          dy = Q["traj"][L, 1] - Q["traj"][L - 1, 1]
          d = jnp.sqrt(dx ** 2 + dy ** 2)
          plt.annotate(
              "",
              (Q["traj"][L - 1, 0], Q["traj"][L - 1, 1]),
              (Q["traj"][L, 0], Q["traj"][L, 1]),
              arrowprops={"arrowstyle": "<-"},
          )
          plt.annotate(
              str(i + 1),
              (Q["traj"][L, 0], Q["traj"][L, 1]),
              xytext=(3, 3),
              textcoords="offset points",
          )
      plt.scatter(
          Q["traj"][L + 1, 0],
          Q["traj"][L + 1, 1],
          c=("red" if jnp.abs(Q["dH"]) > 0.1 else "black"),
          zorder=4,
      )
      #plt.axis('square')
      plt.title(f'L={L}')
      plt.savefig(f'../figures/hmc2d_L{L}.pdf', dpi=300)

make_plot(step=0.03, L=11) # no U-turn
plt.show()
    
make_plot(step=0.03, L=28) # U-turn
plt.show()