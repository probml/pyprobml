# jeffreys prior for bernoulli using 2 paramterizatiobs
# fig 1.9 of 'Bayeysian Modeling and Computation'

import numpy as np
import matplotlib.pyplot as plt 
import pyprobml_utils as pml
import arviz as az
from scipy import stats


np.random.seed(0)

xs = (np.linspace(0, 20, 200), np.linspace(0, 1, 200), np.linspace(-4, 4, 200)) 
dists = (stats.expon(scale=5), stats.beta(0.5, 0.5), stats.norm(0, 1))
fig, ax = plt.subplots(3, 3, figsize=(10,10))
for idx, (dist, x) in enumerate(zip(dists, xs)): 
    draws = dist.rvs(100000)
    data = dist.cdf(draws)
    ax[idx, 0].plot(x, dist.pdf(x)) 
    ax[idx, 1].plot(np.sort(data), np.linspace(0, 1, len(data)))
    az.plot_kde(data, ax=ax[idx, 2])
    if idx==0:
        ax[idx,0].set_title('pdf(X)')
        ax[idx,1].set_title('cdf(Y)')
        ax[idx,2].set_title('pdf(Y)')
        
        
pml.savefig('ecdf_sample.pdf', dpi=300)
plt.show()