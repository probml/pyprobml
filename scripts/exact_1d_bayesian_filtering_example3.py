import functools

from jax import numpy as jnp
from jax import scipy as jsp
from jax import random
import matplotlib.pyplot as plt

import exact_1d_bayesian_filtering_utils as utils


# state transition function
def state_trans_func3(x, v, k=None):
    return 0.7 * x + v


# measurement function
def saturate(x, minval, maxval):
    return jnp.maximum(jnp.minimum(x, maxval), minval)


def measure_func3(x, e, minval=-1.5, maxval=1.5):
    return saturate(x + e, minval=minval, maxval=maxval)


# to get x from measurement without noise
def inv_measure_func3(y):
    return y


# functions to get sample
def v_rvs3(rng_key, shape):
    return random.normal(rng_key, shape=shape)


def e_rvs3(rng_key, shape):
    return random.normal(rng_key, shape=shape) * jnp.sqrt(0.5)


def x0_rvs3(rng_key, shape):
    return random.normal(rng_key, shape=shape) * jnp.sqrt(0.1)


# functions to get density
x0_pdf3 = functools.partial(jsp.stats.norm.pdf, scale=jnp.sqrt(0.1))


# input arguments
rng_key = random.PRNGKey(0)
num_samples = 10000
grid_minval = -6
grid_maxval = 6
num_grid_points = 500
max_iter = 24
iter_ = 18
plot_all_densities = False

rng_key, subkey = random.split(rng_key, num=2)
x_grid, x_true, y = utils.experiment_setup(
    rng_key=rng_key, grid_minval=grid_minval, grid_maxval=grid_maxval,
    num_grid_points=num_grid_points, x0_rvs=x0_rvs3, v_rvs=v_rvs3,
    e_rvs=e_rvs3, f=state_trans_func3, h=measure_func3,
    max_iter=max_iter, plot_xy=True,
)

p_filter, p_pred = utils.novel_density(
    subkey, y, x_grid,
    x0_pdf3, v_rvs3, e_rvs3,
    state_trans_func3, measure_func3, num_samples,
    max_iter, kernel_variance=0.15,
)
p_smooth = None

if plot_all_densities:
    # looking for weird density plot by plotting all max_iter densities
    utils.plot_densities(
        x_true, y, inv_measure_func3,
        x_grid, p_pred, p_filter,
        p_smooth, max_iter,
    )

# plot the kth density
utils.plot_density(
    x_true, y, inv_measure_func3,
    x_grid, p_pred, p_filter,
    p_smooth, k=iter_, legend=True,
    ax=None, title=f"Saturated measurements example densities at $x_{{{iter_}}}$",
)

plt.show()
