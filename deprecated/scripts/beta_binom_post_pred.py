import superimport

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb, beta
from scipy.stats import binom
from scipy import stats

np.random.seed(0)

a_prior, b_prior = 1, 1

Y = stats.bernoulli(0.7).rvs(20)

N1, N0 = Y.sum(), len(Y) - Y.sum()

a_post = a_prior + N1
b_post = b_prior + N0

prior_pred_dist, post_pred_dist = [], []
N = 20

for k in range(N + 1):
    post_pred_dist.append(comb(N, k) * beta(k + a_post, N - k + b_post) / beta(a_post, b_post))
    prior_pred_dist.append(comb(N, k) * beta(k + a_prior, N - k + b_prior) / beta(a_prior, b_prior))

fig, ax = plt.subplots()
ax.bar(np.arange(N + 1), prior_pred_dist, align='center', color='grey')
ax.set_title(f"Prior predictive distribution", fontweight='bold')
ax.set_xlim(-1, 21)
ax.set_xticks(list(range(N + 1)))
ax.set_xticklabels(list(range(N + 1)))
ax.set_ylim(0, 0.15)
ax.set_xlabel("number of success")

fig, ax = plt.subplots()
ax.bar(np.arange(N + 1), post_pred_dist, align='center', color='grey')
ax.set_title(f"Posterior predictive distribution", fontweight='bold')
ax.set_xlim(-1, 21)
ax.set_xticks(list(range(N + 1)))
ax.set_xticklabels(list(range(N + 1)))
ax.set_ylim(0, 0.15)
ax.set_xlabel("number of success")

fig, ax = plt.subplots()
az.plot_dist(np.random.beta(a_prior, b_prior, 10000), plot_kwargs={"color": "0.5"},
             fill_kwargs={'alpha': 1})
ax.set_title("Prior distribution", fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 4)
ax.tick_params(axis='both', pad=7)
ax.set_xlabel("θ")

fig, ax = plt.subplots()
az.plot_dist(np.random.beta(a_post, b_post, 10000), plot_kwargs={"color": "0.5"},
             fill_kwargs={'alpha': 1})
ax.set_title("Posterior distribution", fontweight='bold')
#ax.set_xlim(0, 1)
#ax.set_ylim(0, 4)
ax.tick_params(axis='both', pad=7)
ax.set_xlabel("θ")

plt.show()
