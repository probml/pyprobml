
# Source: chap 2 of 
# https://github.com/aloctavodia/BAP

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
#import seaborn as sns
import pymc3 as pm
import arviz as az


# speed of light data
# Data from http://www.stat.columbia.edu/~gelman/book/data/light.asc
data  = np.array([28, 26, 33, 24, 34, -44, 27, 16, 40, -2,  29,
                  22, 24, 21, 25, 30, 23, 29, 31, 19,
     24, 20, 36, 32, 36, 28, 25, 21, 28, 29,  37, 25, 28, 26, 
     30, 32, 36 ,26, 30, 22,
     36, 23, 27, 27, 28, 27, 31, 27, 26, 33,  26, 32,
     32, 24, 39, 28, 24, 25, 32, 25,
      29, 27, 28, 29, 16, 23]);

np.random.seed(42)
n = 100
mu = 2; sigma = 1;
data = sigma * np.random.randn(n) + mu


with pm.Model() as model_g:
    μ = pm.Uniform('μ', lower=-10, upper=10)
    σ = pm.HalfNormal('σ', sd=10)
    y = pm.Normal('y', mu=μ, sd=σ, observed=data)
    trace_g = pm.sample(1000)

axes = az.plot_joint(trace_g, kind='kde', fill_last=False)
ax = axes[0]
mu = np.mean(data)
sigma = np.std(data)
ax.plot(mu, sigma, 'r*', markersize=12)
plt.savefig('../figures/kde-gauss-2d.pdf', dpi=300)