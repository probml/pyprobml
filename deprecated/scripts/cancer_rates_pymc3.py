

import superimport

import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from collections import defaultdict
import arviz as az

#https://github.com/probml/pmtk3/blob/master/demos/cancerRatesEb.m

data_y = np.array([0, 0, 2, 0, 1, 1, 0, 2, 1, 3, 0, 1, 1, 1, 54, 0, 0, 1, 3, 0]);
data_n = np.array([1083, 855, 3461, 657, 1208, 1025, 527, 1668, 583, 582, 917, 857,
    680, 917, 53637, 874, 395, 581, 588, 383]);
N = len(data_n)

# We put a prior on the mean and precision () of the Beta distribution,
# instead of on the alpha and beta parameters 
with pm.Model() as model_h:
    mu = pm.Beta('mu', 1., 1.)
    kappa = pm.HalfNormal('kappa', 500)
    alpha = pm.Deterministic('alpha', mu*kappa)
    beta = pm.Deterministic('beta', (1.0-mu)*kappa)
    theta = pm.Beta('theta', alpha=alpha, beta=beta, shape=N)
    y = pm.Binomial('y', p=theta, observed=data_y, n=data_n)



np.random.seed(0)
with model_h:
  trace_h = pm.sample(1000, chains=2, cores=1)
  
az.summary(trace_h).round(4)

az.plot_forest(trace_h, var_names=["theta"], combined=True, hdi_prob=0.95);

az.plot_forest(trace_h, var_names=["theta"], combined=True, kind='ridgeplot');

plt.show()