

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


from scipy.stats import poisson

for l in [1.0, 10.0]:
    plt.figure()
    probabilities = poisson.pmf(np.arange(30), l)
    plt.bar(np.arange(30), probabilities)
    plt.xticks(np.arange(0, 30, 5) + 0.4, np.arange(0, 30, 5))
    plt.title(r'$Poi (\lambda = %.2f)$' % l)
    save_fig('poissonPlotDemo_%s.png' % l)
plt.show()
