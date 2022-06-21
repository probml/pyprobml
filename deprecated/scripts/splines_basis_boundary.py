

# Code is from Osvaldo Martin et al, 
# "Bayesian Modeling and Comptuation In Python"
# https://github.com/aloctavodia/BMCP/blob/master/Code/chp_3_5/splines.py

import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from patsy import bs, dmatrix

x = np.linspace(-0.0001, 1, 1000)
knots = [0, 0.2, 0.4, 0.6, 0.8, 1]

_, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, sharey=True)
degrees = [0,1,3]
for i, ax in enumerate(axes):
    deg = degrees[i]
    b_splines = bs(x, degree=deg, knots=knots, lower_bound=-0.01, upper_bound=1.01)
    for b_s in b_splines.T:
        ax.plot(x, b_s, "C3", ls="--")
    ax.plot(x, b_splines[:,deg], lw=2)
    ax.plot(knots, np.zeros_like(knots), "ko", markersize=3)
    for i in range(1, deg+1):
        ax.plot([0, 1], np.array([0, 0])-(i/15), "k.", clip_on=False)
    ax.plot(knots[:deg+2], np.zeros_like(knots[:deg+2]), "C4o", markersize=10)
plt.ylim(0)
plt.xticks([])
plt.yticks([]);
plt.savefig('../figures/splines_basis_boundary.pdf', dpi=300)
