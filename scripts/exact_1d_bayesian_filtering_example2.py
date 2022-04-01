import functools

from jax import scipy as jsp
from jax import random
import matplotlib.pyplot as plt

import exact_1d_bayesian_filtering_utils as utils


# state transition function
def state_trans_func2(x, v, k=None):
    return x + v


# measurement function
def measure_func2(x, e):
    return x + e


# to get x from measurement without noise
def inv_measure_func2(y):
    return y


# functions to get sample
def v_rvs2(rng_key, shape):
    return random.t(rng_key, df=3, shape=shape)


def e_rvs2(rng_key, shape):
    return random.t(rng_key, df=3, shape=shape)


def x0_rvs2(rng_key, shape):
    return random.t(rng_key, df=3, shape=shape)


# functions to get density
pdf2 = functools.partial(jsp.stats.t.pdf, df=3)
v_pdf2 = pdf2
e_pdf2 = pdf2
x0_pdf2 = pdf2


# input arguments
rng_key = random.PRNGKey(0)
grid_minval = -60
grid_maxval = 30
num_grid_points = 500
max_iter = 25
iter_ = 22
plot_all_densities = False

# experiment starts
x_grid, x_true, y = utils.experiment_setup(
    rng_key=rng_key, grid_minval=grid_minval, grid_maxval=grid_maxval,
    num_grid_points=num_grid_points, x0_rvs=x0_rvs2, v_rvs=v_rvs2,
    e_rvs=e_rvs2, f=state_trans_func2, h=measure_func2,
    max_iter=max_iter, plot_xy=False,
)

p_filter, p_pred, p_smooth = utils.point_mass_density(
    y, x_grid, x0_pdf2,
    x_pdf=utils.x_pdf, v_pdf=v_pdf2, e_pdf=e_pdf2,
    f=state_trans_func2, h=measure_func2,
)

if plot_all_densities:
    # looking for weird density plot by plotting all max_iter densities
    utils.plot_densities(
        x_true, y, inv_measure_func2,
        x_grid, p_pred, p_filter,
        p_smooth, max_iter
    )

# plot the kth density
utils.plot_density(
    x_true, y, inv_measure_func2,
    x_grid, p_pred, p_filter,
    p_smooth, k=iter_, legend=True,
    ax=None, title=f"Student's t random walk example densities at $x_{{{iter_}}}$",
)

plt.show()
