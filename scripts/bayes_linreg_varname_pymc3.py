# pymc3 problem with numbers in variable names

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

    
np.random.seed(1)
N = 100
alpha_real = 2.5
beta_real = 0.9
noiseSD = 0.5
eps_real = np.random.normal(0, noiseSD, size=N)

x = np.random.normal(10, 1, N) # centered on 10
y_real = alpha_real + beta_real * x
y = y_real + eps_real


with pm.Model() as model:
    a = pm.Normal('w0', mu=0, sd=10)
    b = pm.Normal('w1', mu=0, sd=1)
    mu = pm.Deterministic('mu', a + b * x)
    y_pred = pm.Normal('y_pred', mu=mu, sd=noiseSD, observed=y)
    trace = pm.sample(1000)

az.plot_trace(trace, var_names=['w0', 'w1'])
