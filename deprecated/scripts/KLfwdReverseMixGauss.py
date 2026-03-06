# Visualize difference between KL(p,q) and KL(q,p) where p is a mix of two
# 2D Gaussians, and q is a single 2D Gaussian
# Author: animesh-007 (fixed/updated)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

mu = np.array([[-1, -1], [1, 1]])

Sigma = np.zeros((2, 2, 2))
Sigma[:, :, 0] = [[1/2, 1/4], [1/4, 1]]
Sigma[:, :, 1] = [[1/2, -1/4], [-1/4, 1]]
SigmaKL = np.array([[3, 2], [2, 3]])

# grid
x = np.arange(-4, 4.1, 0.1)
X, Y = np.meshgrid(x, x)    # X.shape == Y.shape == (len(x), len(x))
pos = np.column_stack([X.ravel(), Y.ravel()])  # (Npoints, 2)

# evaluate pdfs (vectorized)
f1 = multivariate_normal.pdf(pos, mean=mu[0], cov=Sigma[:, :, 0]).reshape(X.shape)
f2 = multivariate_normal.pdf(pos, mean=mu[1], cov=Sigma[:, :, 1]).reshape(X.shape)
klf = multivariate_normal.pdf(pos, mean=[0, 0], cov=SigmaKL).reshape(X.shape)
kll = multivariate_normal.pdf(pos, mean=mu[0], cov=Sigma[:, :, 0] * 0.6).reshape(X.shape)
klr = multivariate_normal.pdf(pos, mean=mu[1], cov=Sigma[:, :, 1] * 0.6).reshape(X.shape)

f = f1 + f2

plots = [klf, kll, klr]

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, plot_ in zip(axs, plots):
    ax.axis('off')
    # mixture contours in blue, comparison distribution contours in red
    ax.contour(X, Y, f, colors='b', zorder=1)
    ax.contour(X, Y, plot_, colors='r', zorder=10)

fig.tight_layout()

out_dir = "../figures"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'klfwdzrevmixgauss.pdf'), dpi=300)
plt.show()
