

# Code is from Osvaldo Martin et al, 
# "Bayesian Modeling and Comptuation In Python"
# https://github.com/aloctavodia/BMCP/blob/master/Code/chp_3_5/splines.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from patsy import bs, dmatrix

x = np.linspace(0., 1., 20)
knots = [0.25, 0.5, 0.75]

B0 = dmatrix("bs(x, knots=knots, degree=0, include_intercept=True) - 1", {"x": x, "knots":knots})
B1 = dmatrix("bs(x, knots=knots, degree=1, include_intercept=True) - 1", {"x": x, "knots":knots})
B3 = dmatrix("bs(x, knots=knots, degree=3, include_intercept=True) - 1", {"x": x, "knots":knots})

_, axes = plt.subplots(1, 3, sharey=True)
for idx, (B, title, ax) in enumerate(zip((B0, B1, B3),
                                     ("Piecewise constant", "Piecewise linear", "Cubic spline"),
                                      axes)):
    ax.imshow(B, cmap="cet_gray_r", aspect="auto")
    ax.set_xticks(np.arange(B.shape[1]))
    ax.set_yticks(np.arange(B.shape[0]))
    ax.set_yticklabels([np.round(v,1) for v in x])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_title(title)
    #for i in range(B.shape[0]):
    #    for j in range(B.shape[1]):
    #        if B[i, j] > 0.01:
    #            text = ax.text(j, i, np.round(B[i, j], 2),
    #                           ha="center", va="center", color="C4", weight="bold")

axes[1].set_xlabel("B-splines")
axes[0].set_ylabel("x", rotation=0, labelpad=15);
plt.savefig('../figures/spline_basis_heatmap.pdf', dpi=300)