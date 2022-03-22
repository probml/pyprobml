import functools

from jax import scipy as jsp
from jax import random
import matplotlib.pyplot as plt

import exact_1d_bayesian_filtering_utils as utils


# state transition function
def f(x, v, k=None):
    return x + v


# measurement function
def h(x, e):
    return x + e


# to get x from measurement without noise
def inv_h(y):
    return y


# functions to get sample
v_rvs = lambda rng_key, shape: random.t(rng_key, df=3, shape=shape)
e_rvs = lambda rng_key, shape: random.t(rng_key, df=3, shape=shape)
x0_rvs = lambda rng_key, shape: random.t(rng_key, df=3, shape=shape)


# functions to get density
pdf = functools.partial(jsp.stats.t.pdf, df=3)
v_pdf = pdf
e_pdf = pdf
x0_pdf = pdf

# input arguments
rng_key = random.PRNGKey(0)
grid_minval = -60
grid_maxval = 30
num_grid_points = 500
K = 25
k = 22

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
    plot_all_densities=False,
    title=f"Student's t random walk example densities at $x_{{{k}}}$",
)
plt.show()
