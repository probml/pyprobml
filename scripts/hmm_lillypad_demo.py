# Example of an HMM
# Author: Gerardo Durán-Martín (@gerdm)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

initial_probs = np.array([0.3, 0.2, 0.5])

# transition matrix
A = np.array([
    [0.3, 0.4, 0.3],
    [0.1, 0.6, 0.3],
    [0.2, 0.3, 0.5]
])

S1 = np.array([
    [1.1, 0],
    [0, 0.3]
])

S2 = np.array([
    [0.3, -0.5],
    [-0.5, 1.3]
])

S3 = np.array([
    [0.8, 0.4],
    [0.4, 0.5]
])


mu_collection = np.array([
    [0.3, 0.3],
    [0.8, 0.5],
    [0.3, 0.8]
])

cov_collection = np.array([S1, S2, S3]) / 60


Xgrid = np.mgrid[0:1:0.01, 0:1.2:0.01]
colors = ["tab:green", "tab:blue", "tab:red"]

np.random.seed(314)
n_samples = 50
z = np.array([0, 1, 2])

# Initial latent variable

def sample_hmm(initial_probs, mu_collection, cov_collection):
    zi = np.random.choice(z, p=initial_probs)
    N = multivariate_normal(mu_collection[zi], cov_collection[zi])
    samples[0] = N.rvs()
    color = colors[zi]
    states_sample = [zi]
    color_sample = [color]

samples = np.zeros((n_samples, 2))

zi = np.random.choice(z, p=initial_probs)
N = multivariate_normal(mu_collection[zi], cov_collection[zi])
samples[0] = N.rvs()
color = colors[zi]
states_sample = [zi]
color_sample = [color]

for i in range(1, n_samples):
    zi = np.random.choice(z, p=A[zi])
    color = colors[zi]
    states_sample.append(zi)
    color_sample.append(color)
    N = multivariate_normal(mu_collection[zi], cov_collection[zi])
    samples[i] = N.rvs()

fig, ax = plt.subplots()
for k, (mu, S, color) in enumerate(zip(mu_collection, cov_collection, colors)):
    N = multivariate_normal(mean=mu, cov=S)
    Z = np.apply_along_axis(N.pdf, 0, Xgrid)
    ax.contour(*Xgrid, Z, levels=[1], colors=color, linewidths=3)
    ax.text(*(mu + 0.13), f"$k$={k + 1}", fontsize=13, horizontalalignment="right")
    
ax.plot(*samples.T, c="black", alpha=0.3, zorder=1)
ax.scatter(*samples.T, c=color_sample, s=30, zorder=2, alpha=0.8)
ax.set_ylim(0, 1.1)


fig, ax = plt.subplots()
ax.step(range(n_samples), states_sample, where="post", c="black", linewidth=1, alpha=0.6)
ax.scatter(range(n_samples), states_sample, c=color_sample, zorder=3)

plt.show()