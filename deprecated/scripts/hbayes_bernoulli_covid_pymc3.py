# From chapter 2 of
# https://github.com/aloctavodia/BAP

import superimport

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
#import seaborn as sns
import pymc3 as pm
import arviz as az

np.random.seed(123)

# Example from BAP
N_samples = np.array([30, 30, 30])
G_samples = np.array([18, 18, 18])  # [3, 3, 3]  [18, 3, 3]


# cancer rates data
G_samples = np.array([0, 0, 2, 0, 1, 1, 0, 2, 1, 3, 0, 1, 1, 1,
                      54, 0, 0, 1, 3, 0])
N_samples = np.array([1083, 855, 3461, 657, 1208, 1025, 527,
                      1668, 583, 582, 917, 857,
    680, 917, 53637, 874, 395, 581, 588, 383])


group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))
    
with pm.Model() as model_h:
    μ = pm.Beta('μ', 1., 1.)
    κ = pm.HalfNormal('κ', 10)

    θ = pm.Beta('θ', alpha=μ*κ, beta=(1.0-μ)*κ, shape=len(N_samples))
    y = pm.Bernoulli('y', p=θ[group_idx], observed=data)

    trace_h = pm.sample(1000, cores=1, chains=2)
    
az.plot_trace(trace_h)
#plt.savefig('B11197_02_20.png', dpi=300)

az.summary(trace_h)



J = len(N_samples)
post_mean = np.zeros(J)
samples = trace_h['θ']
post_mean = np.mean(samples, axis=0)
post_hyper_mean = trace_h['μ'].mean()

mle = G_samples / N_samples
pooled_mle = np.sum(G_samples) / np.sum(N_samples)


axes = az.plot_forest(
    trace_h, var_names='θ', combined=False, colors='cycle')
y_lims = axes[0].get_ylim()
axes[0].vlines(post_hyper_mean, *y_lims)

'''
mle
Out[422]: 
array([0.        , 0.        , 0.00057787, 0.        , 0.00082781,
       0.00097561, 0.        , 0.00119904, 0.00171527, 0.00515464,
       0.        , 0.00116686, 0.00147059, 0.00109051, 0.00100677,
       0.        , 0.        , 0.00172117, 0.00510204, 0.        ])


pooled_mle
Out[424]: 0.0009933126276616582

post_mean
Out[423]: 
array([0.0001577 , 0.00018562, 0.0006269 , 0.0002514 , 0.00092564,
       0.00109622, 0.000321  , 0.00128508, 0.00202207, 0.00539215,
       0.00018146, 0.00135083, 0.00168994, 0.00125866, 0.00101744,
       0.00019823, 0.00045821, 0.00191824, 0.00519503, 0.00043667])

post_hyper_mean
Out[425]: 0.012545970880351854
'''

fig, axs = plt.subplots(4,1, figsize=(8,8))
axs = np.reshape(axs, 4)
xs = np.arange(J)
ax = axs[0]
ax.bar(xs, G_samples)
ax.set_ylim(0, 5)
ax.set_title('number of cases (truncated at 5)')
ax = axs[1]
ax.bar(xs, N_samples)
ax.set_ylim(0, 1000)
ax.set_title('popn size (truncated at 1000)')
ax = axs[2]
ax.bar(xs, mle)
ax.hlines(pooled_mle, 0, J, 'r', lw=3)
ax.set_title('MLE (red line = pooled)')
ax = axs[3]
ax.bar(xs, post_mean)
#ax.hlines(post_hyper_mean, 0, J, 'r', lw=3)
ax.set_title('posterior mean (red line = hparam)')


# Show posterior over hparans
fig, ax= plt.subplots(1,1)
x = np.linspace(0, 1, 100)
for i in np.random.randint(0, len(trace_h), size=100):
    u = trace_h['μ'][i]
    k = trace_h['κ'][i]
    pdf = stats.beta(u*k, (1.0-u)*k).pdf(x)
    ax.plot(x, pdf,  'C1', alpha=0.2)

u_mean = trace_h['μ'].mean()
k_mean = trace_h['κ'].mean()
dist = stats.beta(u_mean*k_mean, (1.0-u_mean)*k_mean)
pdf = dist.pdf(x)
mode = x[np.argmax(pdf)]
mean = dist.moment(1)
ax.plot(x, pdf, lw=3, label=f'mode = {mode:.2f}\nmean = {mean:.2f}')
ax.set_yticks([])
ax.legend()
ax.set_xlabel('$θ_{prior}$')
plt.tight_layout()
#plt.savefig('B11197_02_21.png', dpi=300)

plt.show()