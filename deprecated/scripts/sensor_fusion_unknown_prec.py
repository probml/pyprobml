# Posterior for z
# Author: Aleyna Kara
# This file is translated from sensorFusionUnknownPrec.m

import superimport

import pyprobml_utils as pml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

xs, ys = [1.1, 1.9], [2.9, 4.1]
nx, ny = len(xs), len(ys)

xbar = np.mean(xs)
ybar = np.mean(ys)

sx = np.sum((xs - xbar)**2)/nx
sy = np.sum((ys - ybar)**2)/ny

# MLE
lamx, lamy = 1/sx, 1 /sy
post_prec = (nx * lamx + ny*lamy)
theta = (xbar * nx * lamx + ybar * ny * lamy) / post_prec
post_var = 1/post_prec

# iterate the fixed point iterations
for _ in range(10):
  lamx = nx/np.sum((xs - theta)**2)
  lamy = ny/np.sum((ys - theta)**2)
  theta = (xbar * nx * lamx + ybar * ny* lamy)/(nx * lamx + ny * lamy);

post_var = 1/(nx * lamx + ny * lamy)

start, end, n = -2, 6, 81
grid_theta = np.linspace(start, end, n)

plt.plot(grid_theta, multivariate_normal.pdf(grid_theta, mean=theta, cov=np.sqrt(post_var)), 'b')
plt.xlim([start, end])
plt.ylim(bottom=0)
pml.savefig('sensorFusion2Gauss.pdf', dpi=300)
plt.show()

# Bayesian analysis
fx = (grid_theta - xbar)**2 + sx
fy = (grid_theta - ybar)**2 + sy
post = (1/fx) * (1/fy)

plt.plot(grid_theta, post, 'b')
plt.xlim([start, end])
plt.ylim(bottom=0)
pml.savefig('sensorFusion2Nongauss.pdf', dpi=300)
plt.show()