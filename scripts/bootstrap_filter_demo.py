# Demo of the bootstrap filter under a
# nonlinear discrete system

# Author: Gerardo Gerardo Durán-Martín (@gerdm)

import jax
import nlds_lib as ds
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from jax.scipy import stats
from jax.ops import index_update

def plot_samples(sample_state, sample_obs, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(*sample_state.T, label="state space")
    ax.scatter(*sample_obs.T, s=60, c="tab:green", marker="+")
    ax.scatter(*sample_state[0], c="black", zorder=3)
    ax.legend()
    ax.set_title("Noisy observations from hidden trajectory")
    plt.axis("equal")



plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

def fz(x, dt): return x + dt * jnp.array([jnp.sin(x[1]), jnp.cos(x[0])])
fz_vec = jax.vmap(fz, in_axes=(0, None))
def fx(x): return x

dt = 0.4
nsteps = 100
# Initial state vector
x0 = jnp.array([1.5, 0.0])
# State noise
Qt = jnp.eye(2) * 0.001
# Observed noise
Rt = jnp.eye(2) * 0.05
alpha, beta, kappa = 1, 0, 2

key = random.PRNGKey(314)
model = ds.NLDS(lambda x: fz(x, dt), fx, Qt, Rt)
sample_state, sample_obs = model.sample(key, x0, nsteps)

plot_samples(sample_state, sample_obs)

particle_filter = ds.BootstrapFiltering(lambda x: fz_vec(x, dt), fx, Qt, Rt)
key = random.PRNGKey(314)
pf_mean = particle_filter.filter(key, x0, sample_obs)

plt.plot(*pf_mean.T)
plt.show()