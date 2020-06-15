# 8 schools model

#https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html


import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from collections import defaultdict
import arviz as az


print('Runing on PyMC3 v{}'.format(pm.__version__))

# Data of the Eight Schools Model
J = 8
y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])

names=[]; 
for t in range(8):
    names.append('theta {}'.format(t)); 
print(names)

# Plot raw data
fig, ax = plt.subplots()
y_pos = np.arange(8)
ax.errorbar(y,y_pos, xerr=sigma, fmt='o')
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.invert_yaxis()  # labels read top-to-bottom
plt.show()



# Centered model
with pm.Model() as Centered_eight:
    mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=5)
    sigma_alpha = pm.HalfCauchy('sigma_alpha', beta=5)
    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=J)
    obs = pm.Normal('obs', mu=alpha, sigma=sigma, observed=y)
    
    
np.random.seed(0)
with Centered_eight:
    trace_centered = pm.sample(1000, chains=4)
    
pm.summary(trace_centered).round(2)
# Effective sample size is << 4*1000, especially for tau
# Also, PyMC3 gives various warnings about not mixing

# Display the total number and percentage of divergent chains
diverging = trace_centered['diverging']
print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
diverging_pct = diverging.nonzero()[0].size / len(trace_centered) * 100
print('Percentage of Divergent Chains: {:.1f}'.format(diverging_pct))

az.plot_autocorr(trace_centered, var_names=['mu_alpha', 'sigma_alpha']);
 
az.plot_forest(trace_centered, var_names="alpha", credible_interval=0.95);
    

# Non-Centered model

with pm.Model() as NonCentered_eight:
    mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=5)
    sigma_alpha = pm.HalfCauchy('sigma_alpha', beta=5)
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=J)
    alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset)
    #alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=J)
    obs = pm.Normal('obs', mu=alpha, sigma=sigma, observed=y)
    


    
np.random.seed(0)
with NonCentered_eight:
    trace_noncentered = pm.sample(1000, chains=4)
    
pm.summary(trace_noncentered).round(2)
# Things look much beteter: r_hat = 1, ESS ~ 4*1000

az.plot_autocorr(trace_noncentered, var_names=['mu', 'tau']);


az.plot_forest(trace_noncentered, var_names="alpha",
               combined=True, credible_interval=0.95);




# Plot the "funnel of hell"
# Based on
# https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/GLM_hierarchical_non_centered.ipynb

for group in range(J):
  #x = pd.Series(trace_centered['alpha'][:, group], name=f'alpha {group}')
  #y = pd.Series(trace_centered['sigma_alpha'], name='sigma_alpha')
  #sns.jointplot(x, y);

  fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
  x = pd.Series(trace_centered['alpha'][:, group], name=f'alpha {group}')
  y  = pd.Series(trace_centered['sigma_alpha'], name='sigma_alpha')
  axs[0].plot(x, y, '.');
  axs[0].set(title='Centered', ylabel='sigma_alpha', xlabel=f'alpha {group}')
  x = pd.Series(trace_noncentered['alpha'][:, group], name=f'alpha {group}')
  y  = pd.Series(trace_noncentered['sigma_alpha'], name='sigma_alpha')
  axs[1].plot(x, y, '.');
  axs[1].set(title='Non-centered', ylabel='sigma_alpha', xlabel=f'alpha {group}')