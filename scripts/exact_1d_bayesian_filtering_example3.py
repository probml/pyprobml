import functools

from jax import numpy as jnp
from jax import scipy as jsp
from jax import random
import matplotlib.pyplot as plt

import exact_1d_bayesian_filtering_utils as utils


# state transition function
def f(x, v, k=None):
    return 0.7 * x + v


# measurement function
def sat(x, minval, maxval):
    return jnp.maximum(jnp.minimum(x, maxval), minval)


def h(x, e):
    return sat(x + e, minval=-1.5, maxval=1.5)


# to get x from measurement without noise
def inv_h(y):
    return y


# functions to get sample
v_rvs = lambda rng_key, shape: random.normal(rng_key, shape=shape)
e_rvs = lambda rng_key, shape: random.normal(rng_key, shape=shape) * jnp.sqrt(0.5)
x0_rvs = lambda rng_key, shape: random.normal(rng_key, shape=shape) * jnp.sqrt(0.1)


# functions to get density
x0_pdf = functools.partial(jsp.stats.norm.pdf, scale=jnp.sqrt(0.1))

# input arguments
rng_key = random.PRNGKey(0)
num_samples = 10000
grid_minval = -6
grid_maxval = 6
num_grid_points = 500
K = 24
k = 18

key, subkey = random.split(rng_key, num=2)
x_grid, x_true, y = utils.experiment_setup(
    rng_key=key,
    grid_minval=grid_minval,
    grid_maxval=grid_maxval,
    num_grid_points=num_grid_points,
    x0_rvs=x0_rvs,
    v_rvs=v_rvs,
    e_rvs=e_rvs,
    f=f,
    h=h,
    K=K,
    plot_xy=True,
)
utils.novel_experiment(
    rng_key=subkey,
    x_grid=x_grid,
    x_true=x_true,
    y=y,
    x0_pdf=x0_pdf,
    v_rvs=v_rvs,
    e_rvs=e_rvs,
    f=f,
    h=h,
    inv_h=inv_h,
    K=K,
    k=k,
    num_samples=num_samples,
    plot_all_densities=False,
    title=f"Saturated measurements example densities at $x_{{{k}}}$",
)
plt.show()
