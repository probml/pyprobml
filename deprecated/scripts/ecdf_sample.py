# empirical cdf 
# fig 11.17 of 'Bayeysian Modeling and Computation'

import superimport

import numpy as np
import matplotlib.pyplot as plt 
import pyprobml_utils as pml
import arviz as az
from scipy import stats


import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
seaborn.set()
seaborn.set_style("whitegrid")

# Font sizes
SIZE_SMALL = 14
SIZE_MEDIUM = 18
SIZE_LARGE = 24

# https://stackoverflow.com/a/39566040
plt.rc('font', size=SIZE_SMALL)          # controls default text sizes
plt.rc('axes', titlesize=SIZE_SMALL)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE_SMALL)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE_SMALL)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE_SMALL)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE_SMALL)    # legend fontsize  
plt.rc('figure', titlesize=SIZE_LARGE)   # fontsize of the figure title

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
        
plt.tight_layout()       
pml.savefig('ecdf_sample.pdf', dpi=300)
plt.show()

for idx, (dist, x) in enumerate(zip(dists, xs)): 
    draws = dist.rvs(100000)
    data = dist.cdf(draws)
    plt.figure()
    plt.plot(x, dist.pdf(x)) 
    if idx==0: plt.title('pdf(X)')
    pml.savefig(f'ecdf_{idx}_pdfX.pdf', dpi=300)
        
    plt.figure()
    plt.plot(np.sort(data), np.linspace(0, 1, len(data)))
    if idx==0: plt.title('cdf(Y)')
    pml.savefig(f'ecdf_{idx}_cdfY.pdf', dpi=300)
    
    fig, ax = plt.subplots()
    az.plot_kde(data, ax=ax)
    if idx==0: plt.title('pdf(Y)')
    pml.savefig(f'ecdf_{idx}_pdfY.pdf', dpi=300)