# 8 schools model

#https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html


import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('seaborn-darkgrid')
from collections import defaultdict
import arviz as az


print('Runing on PyMC3 v{}'.format(pm.__version__))

# Data of the Eight Schools Model
J = 8
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])
sigma2 = np.power(sigma, 2)

#pooledMLE = np.mean(y)
pooledMLE = np.sum(y / sigma2) / np.sum(1/sigma2) # 7.7

names=[]; 
for t in range(8):
    names.append('theta {}'.format(t)); 
print(names)

# Plot raw data
fig, ax = plt.subplots()
y_pos = np.arange(J)
ax.errorbar(y, y_pos, xerr=sigma, fmt='o')
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.axvline(pooledMLE, color='r', linestyle='-')
plt.savefig('../figures/hbayes_schools8_data.pdf', dpi=300)
plt.show()


# Centered model
with pm.Model() as Centered_eight:
    mu = pm.Normal('mu', mu=0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
    
    
np.random.seed(0)
with Centered_eight:
    trace_centered = pm.sample(1000, chains=4, cores=1)
    
print(pm.summary(trace_centered).round(2))
# Effective sample size is << 4*1000, especially for tau
# Also, PyMC3 gives various warnings about not mixing

# Display the total number and percentage of divergent chains
diverging = trace_centered['diverging']
print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
diverging_pct = diverging.nonzero()[0].size / len(trace_centered) * 100
print('Percentage of Divergent Chains: {:.1f}'.format(diverging_pct))

az.plot_autocorr(trace_centered, var_names=['mu', 'tau']);
 
az.plot_forest(trace_centered, var_names="theta", credible_interval=0.95);
    

# Non-Centered model

with pm.Model() as NonCentered_eight:
    mu = pm.Normal('mu', mu=0, sigma=10)
    tau = pm.HalfCauchy('tau', beta=5)
    theta_offset = pm.Normal('theta_offset', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_offset)
    #theta = pm.Normal('theta', mu=mu, sigma=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
    

    
np.random.seed(0)
with NonCentered_eight:
    trace_noncentered = pm.sample(1000, chains=4)
    
print(pm.summary(trace_noncentered).round(2))
# Things look much beteter: r_hat = 1, ESS ~ 4*1000

az.plot_autocorr(trace_noncentered, var_names=['mu', 'tau']);

post_mean = np.mean(trace_noncentered['theta'])
hyper_mean = np.mean(trace_noncentered['mu']) # 4.340815448509472

axes = az.plot_forest(trace_noncentered, var_names="theta",
               combined=True, credible_interval=0.95);
y_lims = axes[0].get_ylim()
axes[0].vlines(hyper_mean, *y_lims)
plt.savefig('../figures/hbayes_schools8_forest.pdf', dpi=300)


az.plot_posterior(trace_noncentered['tau'],
                  credible_interval=0.95)
plt.savefig('../figures/hbayes_schools8_tau.pdf', dpi=300)


# Plot the "funnel of hell"
# Based on
# https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/GLM_hierarchical_non_centered.ipynb

for group in range(J):
  #x = pd.Series(trace_centered['theta'][:, group], name=f'theta {group}')
  #y = pd.Series(trace_centered['tau'], name='tau')
  #sns.jointplot(x, y);

  fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
  x = pd.Series(trace_centered['theta'][:, group], name=f'theta {group}')
  y  = pd.Series(trace_centered['tau'], name='tau')
  axs[0].plot(x, y, '.');
  axs[0].set(title='Centered', ylabel='tau', xlabel=f'theta {group}')
  x = pd.Series(trace_noncentered['theta'][:, group], name=f'theta {group}')
  y  = pd.Series(trace_noncentered['tau'], name='tau')
  axs[1].plot(x, y, '.');
  axs[1].set(title='Non-centered', ylabel='tau', xlabel=f'theta {group}')
  
  