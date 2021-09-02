# Gaussian process binary classification in 1d
# Code is based on 
#https://github.com/aloctavodia/BAP/blob/master/code/Chp7/07_Gaussian%20process.ipynb

import superimport

import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit as logistic
import pyprobml_utils as pml

import matplotlib.pyplot as plt
import arviz as az

from sklearn.datasets import load_iris
    
url = 'https://github.com/aloctavodia/BAP/blob/master/code/data/space_flu.csv?raw=true'

df_sf = pd.read_csv(url)
age = df_sf.age.values[:, None]
space_flu = df_sf.space_flu

ax = df_sf.plot.scatter('age', 'space_flu', figsize=(8, 5))
ax.set_yticks([0, 1])
ax.set_yticklabels(['healthy', 'sick'])
pml.savefig('space_flu.pdf', bbox_inches='tight')


with pm.Model() as model_space_flu:
    ℓ = pm.HalfCauchy('ℓ', 1)
    cov = pm.gp.cov.ExpQuad(1, ℓ) + pm.gp.cov.WhiteNoise(1E-5)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior('f', X=age)
    y_ = pm.Bernoulli('y', p=pm.math.sigmoid(f), observed=space_flu)
    trace_space_flu = pm.sample(1000, chains=1, cores=1, compute_convergence_checks=False)
    
    
X_new = np.linspace(0, 80, 200)[:, None]

with model_space_flu:
    f_pred = gp.conditional('f_pred', X_new)
    pred_samples = pm.sample_posterior_predictive(trace_space_flu,
                                                  var_names=['f_pred'],
                                                  samples=1000)
    
_, ax = plt.subplots(figsize=(10, 6))

fp = logistic(pred_samples['f_pred'])
fp_mean = np.nanmean(fp, 0)

ax.scatter(age, np.random.normal(space_flu, 0.02),
           marker='.', color=[f'C{ci}' for ci in space_flu])

ax.plot(X_new[:, 0], fp_mean, 'C2', lw=3)

az.plot_hdi(X_new[:, 0], fp, color='C2')
ax.set_yticks([0, 1])
ax.set_yticklabels(['healthy', 'sick'])
ax.set_xlabel('age')
pml.savefig('gp_classify_spaceflu.pdf', dpi=300)
