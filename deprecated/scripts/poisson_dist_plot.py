import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
import pyprobml_utils as pml
from scipy.stats import poisson

for l in [1.0, 10.0]:
    plt.figure()
    probabilities = poisson.pmf(np.arange(30), l)
    plt.bar(np.arange(30), probabilities)
    plt.xticks(np.arange(0, 30, 5) + 0.4, np.arange(0, 30, 5))
    plt.title(r'$Poi (\lambda = %.2f)$' % l)
    pml.savefig('poissonPlotDemo_%s.png' % l)
plt.show()
