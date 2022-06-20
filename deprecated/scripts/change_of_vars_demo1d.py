import superimport

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


xs = np.linspace(-1,1,21)
a = -1
b = 1
px = 1/(b-a) * np.ones(len(xs))

fn = lambda x: x**2
ys = fn(xs)

#analytic
ppy = 1/(2*np.sqrt(ys))

#monte carlo
n = 1000
np.random.seed(42)
samples = np.random.uniform(a,b, size=n)
samples2 = fn(samples)

print(np.mean(samples2))

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].plot(xs, px, "-")
ax[1].plot(ys, ppy, "-")
sns.distplot(samples2, kde=False, ax=ax[2], bins=20, norm_hist=True, hist_kws=dict(edgecolor="k", linewidth=0.5))
plt.savefig("../figures/changeOfVars.pdf")
plt.show()
