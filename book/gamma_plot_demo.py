#!/usr/bin/env python

# Plot the gamma distribution

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma
from utils import save_fig

x = np.linspace(0, 7, 100)
for a in [1, 1.5, 2]:
    y = gamma.pdf(x, a)
    plt.plot(x, y)
plt.legend(['a=%.1f' % a for a in [1, 1.5, 2]])
save_fig('gammaPlotDemo.pdf')
plt.show()

