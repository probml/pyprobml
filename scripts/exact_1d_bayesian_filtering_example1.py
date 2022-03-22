import functools

from jax import numpy as jnp
from jax import scipy as jsp
from jax import random
import matplotlib.pyplot as plt

import exact_1d_bayesian_filtering_utils as utils


# state transition function
def f(x, v, k):
    return x / 2 + 25 * x / (1 + x**2) + 8 * jnp.cos(1.2 * (k + 1)) + v


# measurement function
def h(x, e):
    return x**2 / 20 + e


# to get x from measurement without noise
def inv_h(y):
    x = jnp.sqrt(20 * y)
    return [x, -x]


# functions to get sample
v_rvs = lambda rng_key, shape: random.normal(rng_key, shape=shape) * jnp.sqrt(10)
e_rvs = lambda rng_key, shape: random.normal(rng_key, shape=shape)
x0_rvs = lambda rng_key, shape: random.normal(rng_key, shape=shape)


# functions to get density
v_pdf = functools.partial(jsp.stats.norm.pdf, scale=jnp.sqrt(10))
e_pdf = functools.partial(jsp.stats.norm.pdf, scale=1)
x0_pdf = jsp.stats.norm.pdf


# input arguments
rng_key = random.PRNGKey(4)
grid_minval = -30
grid_maxval = 30
num_grid_points = 500
K = 20
k = 14

# experiment starts
x_grid, x_true, y = utils.experiment_setup(
    rng_key=rng_key,
    grid_minval=grid_minval,
    grid_maxval=grid_maxval,
    num_grid_points=num_grid_points,
    x0_rvs=x0_rvs,
    v_rvs=v_rvs,
    e_rvs=e_rvs,
    f=f,
    h=h,
    K=K,
    plot_xy=False,
)

utils.point_mass_experiment(
    x_grid=x_grid,
    x_true=x_true,
    y=y,
    x0_pdf=x0_pdf,
    x_pdf=utils.x_pdf,
    v_pdf=v_pdf,
    e_pdf=e_pdf,
    f=f,
    h=h,
    inv_h=inv_h,
    K=K,
    k=k,
    plot_all_densities=True,
    title=f"Particle filter example densities at $x_{{{k}}}$",
)
plt.show()
