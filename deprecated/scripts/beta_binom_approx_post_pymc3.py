# 1d approixmation to beta binomial model
# https://github.com/aloctavodia/BAP


import superimport

import pymc3 as pm
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import math
import pyprobml_utils as pml

#data = np.repeat([0, 1], (10, 3))
data = np.repeat([0, 1], (10, 1))
h = data.sum()
t = len(data) - h

# Exact

plt.figure()
x = np.linspace(0, 1, 100)
xs = x #grid
dx_exact = xs[1]-xs[0]
post_exact = stats.beta.pdf(xs, h+1, t+1)
post_exact = post_exact / np.sum(post_exact)
plt.plot(xs, post_exact)
plt.yticks([])
plt.title('exact posterior')
pml.savefig('bb_exact.pdf')


# Grid 
def posterior_grid(heads, tails, grid_points=100):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)  # uniform prior
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    #posterior = posterior * grid_points
    return grid, posterior


n = 20
grid, posterior = posterior_grid(h, t, n) 
dx_grid = grid[1] - grid[0]
sf = dx_grid / dx_exact # Jacobian scale factor
plt.figure()
#plt.stem(grid, posterior, use_line_collection=True)
plt.bar(grid, posterior, width=1/n, alpha=0.2)
plt.plot(xs, post_exact*sf)
plt.title('grid approximation')
plt.yticks([])
plt.xlabel('θ');
pml.savefig('bb_grid.pdf')


# Laplace
with pm.Model() as normal_aproximation:
    theta = pm.Beta('theta', 1., 1.)
    y = pm.Binomial('y', n=1, p=theta, observed=data) # Bernoulli
    mean_q = pm.find_MAP()
    std_q = ((1/pm.find_hessian(mean_q, vars=[theta]))**0.5)[0]
    mu = mean_q['theta']

print([mu, std_q])

plt.figure()
plt.plot(xs, stats.norm.pdf(xs, mu, std_q), '--', label='Laplace')
post_exact = stats.beta.pdf(xs, h+1, t+1)
plt.plot(xs, post_exact, label='exact')
plt.title('Quadratic approximation')
plt.xlabel('θ', fontsize=14)
plt.yticks([])
plt.legend()
pml.savefig('bb_laplace.pdf');



# HMC
with pm.Model() as hmc_model:
    theta = pm.Beta('theta', 1., 1.)
    y = pm.Binomial('y', n=1, p=theta, observed=data) # Bernoulli
    trace = pm.sample(1000, random_seed=42, cores=1, chains=2)
thetas = trace['theta']
axes = az.plot_posterior(thetas, hdi_prob=0.95)
pml.savefig('bb_hmc.pdf');

az.plot_trace(trace)
pml.savefig('bb_hmc_trace.pdf', dpi=300)

# ADVI
with pm.Model() as mf_model:
    theta = pm.Beta('theta', 1., 1.)
    y = pm.Binomial('y', n=1, p=theta, observed=data) # Bernoulli
    mean_field = pm.fit(method='advi')
    trace_mf = mean_field.sample(1000)
thetas = trace_mf['theta']
axes = az.plot_posterior(thetas, hdi_prob=0.95)
pml.savefig('bb_mf.pdf');

plt.show()


# track mean and std
with pm.Model() as mf_model:
    theta = pm.Beta('theta', 1., 1.)
    y = pm.Binomial('y', n=1, p=theta, observed=data) # Bernoulli
    advi = pm.ADVI()
    tracker = pm.callbacks.Tracker(
        mean=advi.approx.mean.eval,  # callable that returns mean
        std=advi.approx.std.eval  # callable that returns std
        )  
    approx = advi.fit(callbacks=[tracker])

trace_approx = approx.sample(1000)
thetas = trace_approx['theta']

plt.figure()
plt.plot(tracker['mean'])
plt.title('Mean')
pml.savefig('bb_mf_mean.pdf');

plt.figure()
plt.plot(tracker['std'])
plt.title('Std ')
pml.savefig('bb_mf_std.pdf');

plt.figure()
plt.plot(advi.hist)
plt.title('Negative ELBO');
pml.savefig('bb_mf_elbo.pdf');

plt.figure()
sns.kdeplot(thetas);
plt.title('KDE of posterior samples')
pml.savefig('bb_mf_kde.pdf');


fig,axs = plt.subplots(1,4, figsize=(30,10))
mu_ax = axs[0]
std_ax = axs[1]
elbo_ax = axs[2]
kde_ax = axs[3] 
mu_ax.plot(tracker['mean'])
mu_ax.set_title('Mean')
std_ax.plot(tracker['std'])
std_ax.set_title('Std ')
elbo_ax.plot(advi.hist)
elbo_ax.set_title('Negative ELBO');
kde_ax = sns.kdeplot(thetas);
kde_ax.set_title('KDE of posterior samples')
pml.savefig('bb_mf_panel.pdf');


fig = plt.figure(figsize=(16, 9))
mu_ax = fig.add_subplot(221)
std_ax = fig.add_subplot(222)
hist_ax = fig.add_subplot(212)
mu_ax.plot(tracker['mean'])
mu_ax.set_title('Mean track')
std_ax.plot(tracker['std'])
std_ax.set_title('Std track')
hist_ax.plot(advi.hist)
hist_ax.set_title('Negative ELBO track');
pml.savefig('bb_mf_tracker.pdf');

trace_approx = approx.sample(1000)
thetas = trace_approx['theta']
axes = az.plot_posterior(thetas, hdi_prob=0.95)

plt.show()