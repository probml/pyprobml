# From chapter 2 of
# https://github.com/aloctavodia/BAP

import superimport

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
import arviz as az

np.random.seed(123)

# Example from BAP
N_samples = np.array([30, 30, 30])
G_samples = np.array([18, 18, 18])  # [3, 3, 3]  [18, 3, 3]





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

axes = az.plot_forest(
    trace_h, var_names='θ', combined=True, colors='cycle',
    kind='ridgeplot')



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