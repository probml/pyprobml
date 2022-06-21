import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pyprobml_utils as pml


np.random.seed(0)
n = 500
min_x, max_x = -5, 5
min_y, max_y = -5, 5
x, z = np.meshgrid(np.linspace(min_x, max_x, n), np.linspace(min_y, max_y, n))
logp = norm.logpdf(z, 0, 3) + norm.logpdf(x, 0, np.exp(z / 2))

vmin = np.percentile(logp.flatten(), 42)

logp = np.where(logp < vmin, np.NaN, logp)

n_ticks, n_colors = 5, 6

fig = plt.figure(figsize=(8, 8))
ax = fig.gca()
ax.set_axisbelow(True)
ax.set_facecolor("#EAEBF0")
ax.grid(color='white', linestyle='-', linewidth=3, )
ax.imshow(logp, cmap=matplotlib.cm.get_cmap("viridis_r", n_colors), extent=[min_x, max_x, min_y, max_y],
          origin="lower").set_zorder(1)
pml.savefig('neals-funnel.pdf')
plt.show()