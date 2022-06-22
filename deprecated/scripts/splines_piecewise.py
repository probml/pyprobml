# Plot piecewise spline curves
# Code is from Osvaldo Martin et al, 
# "Bayesian Modeling and Comptuation In Python"
# https://github.com/aloctavodia/BMCP/blob/master/Code/chp_3_5/splines.py

import superimport

import numpy as np
import matplotlib.pyplot as plt

# Convenience function to compute the β cofficients
def ols(X, y):
    """Compute ordinary least squares in closed-form"""
    β = np.linalg.solve(X.T @ X, X.T @ y)
    return X @ β


# Define the knots
knots = 1.57, 4.71

# Get x-y values from true function
x_min, x_max = 0, 6
x_true = np.linspace(x_min, x_max, 200)
y_true = np.sin(x_true)

# Split the x-y values into 3 intervals defined by the knots
x_intervals = [x_true[x_true <= knots[0]],
               x_true[(knots[0] < x_true) & (x_true < knots[1])],
               x_true[x_true >= knots[1]],
               ]

y_intervals = [y_true[x_true <= knots[0]],
               y_true[(knots[0] < x_true) & (x_true < knots[1])],
               y_true[x_true >= knots[1]],
               ]

# Prepare figure
_, ax = plt.subplots(2, 2, figsize=(9, 6), 
                     constrained_layout=True, sharex=True, sharey=True)
ax = np.ravel(ax)

# Plot the knots and true function
for ax_ in ax:
    ax_.vlines(knots, -1, 1, color='grey', ls='--')
    ax_.plot(x_true, y_true, 'C4', lw=4, alpha=0.5)
    ax_.set_xticks([])
    ax_.set_yticks([])

# Compute and plot pointwise step function
ax[0].hlines(y_intervals[0].mean(), color='k', xmin=x_min, xmax=knots[0])
ax[0].hlines(y_intervals[1].mean(), color='k', xmin=knots[0], xmax=knots[1])
ax[0].hlines(y_intervals[2].mean(), color='k', xmin=knots[1], xmax=x_max)
ax[0].set_title('Piecewise Constant')

# Compute and plot pointwise linear function
B = np.array(
    [np.ones_like(x_true),
     x_true,
     np.where(x_true < knots[0], 0, (x_true - knots[0])),
     np.where(x_true < knots[1], 0, (x_true - knots[1])),
     ]).T

y_hat = ols(B, y_true)
ax[1].plot(x_true, y_hat, c='k')
ax[1].set_title('Piecewise Linear')

# Compute and plot pointwise quadratic function
B = np.array(
    [np.ones_like(x_true),
     x_true,
     x_true ** 2,
     np.where(x_true < knots[0], 0, (x_true - knots[0]) ** 2),
     np.where(x_true < knots[1], 0, (x_true - knots[1]) ** 2),
     ]).T

y_hat = ols(B, y_true)
ax[2].set_title('Piecewise Quadratic')
ax[2].plot(x_true, y_hat, color='k')

# Compute and plot pointwise cubic function
B = np.array(
    [np.ones_like(x_true),
     x_true,
     x_true ** 2,
     x_true ** 3,
     np.where(x_true < knots[0], 0, (x_true - knots[0]) ** 3),
     np.where(x_true < knots[1], 0, (x_true - knots[1]) ** 3),
     ]).T

y_hat = ols(B, y_true)
ax[3].set_title('Piecewise Cubic')
ax[3].plot(x_true, y_hat, color='k')
plt.savefig("../figures/splines_piecewise_curves.pdf", dpi=300)
