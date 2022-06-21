# Plot pdf and cdf of standard normal

import superimport

import numpy as np
import matplotlib.pyplot as plt


import pyprobml_utils as pml

from scipy.stats import norm


X = np.linspace(-3, 3, 500)
rv = norm(0, 1)
fig, ax = plt.subplots()
ax.plot(X, rv.pdf(X))
plt.title("Gaussian pdf")
pml.save_fig("gaussian1d.pdf")
plt.show()

fig, ax = plt.subplots()
ax.plot(X, rv.cdf(X))
plt.title("Gaussian cdf")
pml.save_fig("gaussianCdf.pdf")
plt.show()
