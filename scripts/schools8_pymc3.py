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
# tau = 25.

# Centered model
with pm.Model() as Centered_eight:
    mu = pm.Normal('mu', mu=0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
    
np.random.seed(0)
with Centered_eight:
    trace_centered = pm.sample(1000, chains=4)
    
pm.summary(trace_centered).round(2)
# Effective sample size is << 4*1000, especially for tau
# Also, PyMC3 gives various warnings about not mixing

           
az.plot_autocorr(trace_centered, var_names=['mu', 'tau']);

#az.plot_forest(trace_centered, var_names="theta", credible_interval=0.95);
    

# Non-Centered model

with pm.Model() as NonCentered_eight:
    mu = pm.Normal('mu', mu=0, sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)
    eta = pm.Normal('eta', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * eta)
    obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)
    
np.random.seed(0)
with NonCentered_eight:
    trace_noncentered = pm.sample(1000, chains=4)
    
pm.summary(trace_noncentered).round(2)
# Things look much beteter: r_hat = 1, ESS ~ 4*1000

az.plot_autocorr(trace_noncentered, var_names=['mu', 'tau']);


data = az.from_pymc3(
    trace=trace_noncentered,
    model=NonCentered_eight,
)

names=[]; 
for t in range(8):
    names.append('theta {}'.format(t)); 
print(names)
 
az.plot_forest(trace_noncentered, var_names="theta",
               combined=True, credible_interval=0.95);


# Plot raw data
fig, ax = plt.subplots()
y_pos = np.arange(8)
ax.errorbar(y,y_pos, xerr=sigma, fmt='o')
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.invert_yaxis()  # labels read top-to-bottom
plt.show()

