import functools

from jax import numpy as jnp
from jax import scipy as jsp
from jax import random
import matplotlib.pyplot as plt

import exact_1d_bayesian_filtering_utils as utils


# state transition function
def state_trans_func1(x, v, k):
    return x / 2 + 25 * x / (1 + x**2) + 8 * jnp.cos(1.2 * (k + 1)) + v


# measurement function
def measure_func1(x, e):
    return x**2 / 20 + e


# to get x from measurement without noise
def inv_measure_func1(y):
    x = jnp.sqrt(20 * y)
    return [x, -x]


# functions to get sample
def v_rvs1(rng_key, shape):
    return random.normal(rng_key, shape=shape) * jnp.sqrt(10)


def e_rvs1(rng_key, shape):
    return random.normal(rng_key, shape=shape)


def x0_rvs1(rng_key, shape):
    return random.normal(rng_key, shape=shape)


# functions to get density
v_pdf1 = functools.partial(jsp.stats.norm.pdf, scale=jnp.sqrt(10))
e_pdf1 = functools.partial(jsp.stats.norm.pdf, scale=1)
x0_pdf1 = jsp.stats.norm.pdf


# input arguments
rng_key = random.PRNGKey(4)
grid_minval = -30
grid_maxval = 30
num_grid_points = 500
max_iter = 20
iter_ = 14
plot_all_densities = False


# experiment starts
x_grid, x_true, y = utils.experiment_setup(
    rng_key=rng_key, grid_minval=grid_minval, grid_maxval=grid_maxval,
    num_grid_points=num_grid_points, x0_rvs=x0_rvs1, v_rvs=v_rvs1,
    e_rvs=e_rvs1, f=state_trans_func1, h=measure_func1,
    max_iter=max_iter, plot_xy=False,
)


p_filter, p_pred, p_smooth = utils.point_mass_density(
    y, x_grid, x0_pdf1,
    x_pdf=utils.x_pdf, v_pdf=v_pdf1, e_pdf=e_pdf1,
    f=state_trans_func1, h=measure_func1,
)

if plot_all_densities:
    # looking for weird density plot by plotting all max_iter densities
    utils.plot_densities(
        x_true, y, inv_measure_func1,
        x_grid, p_pred, p_filter,
        p_smooth, max_iter,
    )

# plot the kth density
utils.plot_density(
    x_true, y, inv_measure_func1,
    x_grid, p_pred, p_filter,
    p_smooth, k=iter_, legend=True,
    ax=None, title=f"Particle filter example densities at $x_{{{iter_}}}$",
)

plt.show()
