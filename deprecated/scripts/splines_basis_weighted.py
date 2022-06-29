

# Code is from Osvaldo Martin et al, 
# "Bayesian Modeling and Comptuation In Python"
# https://github.com/aloctavodia/BMCP/blob/master/Code/chp_3_5/splines.py

import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from patsy import bs, dmatrix
import pyprobml_utils as pml

x = np.linspace(0., 1., 500)
knots = [0.25, 0.5, 0.75]

B0 = dmatrix("bs(x, knots=knots, degree=0, include_intercept=True) - 1", {"x": x, "knots":knots})
B1 = dmatrix("bs(x, knots=knots, degree=1, include_intercept=True) - 1", {"x": x, "knots":knots})
B3 = dmatrix("bs(x, knots=knots, degree=3, include_intercept=True) - 1", {"x": x, "knots":knots})

np.random.seed(1563)
_, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True,
                     sharey='row')
for idx, (B, title) in enumerate(zip((B0, B1, B3),
                                     ("Piecewise constant", "Piecewise linear", "Cubic spline"))):
    # plot spline basis functions
    for i in range(B.shape[1]):
        ax[0, idx].plot(x, B[:, i], color=str(1-(i+1)/B.shape[1]), lw=2, ls="--")
    # we generate some positive random coefficients (there is nothing wrong with negative values)
    β = np.abs(np.random.normal(0, 1, size=B.shape[1]))
    # plot spline basis functions scaled by its β
    for i in range(B.shape[1]):
        ax[1, idx].plot(x, B[:, i]*β[i],
                        color=str(1-(i+1)/B.shape[1]), lw=2, ls="--")
    # plot the sum of the basis functions
    ax[1, idx].plot(x, np.dot(B, β), color='k', lw=3)
    # plot the knots
    ax[0, idx].plot(knots, np.zeros_like(knots), "ko")
    ax[1, idx].plot(knots, np.zeros_like(knots), "ko")
    ax[0, idx].set_title(title)

pml.savefig('splines_weighted.pdf', dpi=300)
plt.show()