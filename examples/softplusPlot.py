#!/usr/bin/env python

# Plots the softplus function.

import matplotlib.pyplot as plt
import numpy as np
import os
from utils.util import save_fig

fig, ax = plt.subplots()

x = np.linspace(-4, 4, 1000)
y = np.log(1+np.exp(x))
ax.plot(x, y, 'r:', label='softplus', linewidth=3)

y = np.maximum(0, x)
ax.plot(x, y, 'b-', label='relu', linewidth=2)

legend = ax.legend(loc='upper center')

plt.ylim([-0.1, 4.5])
#plt.title('nonlinear functions')
save_fig('softplusPlot.pdf')
plt.show()

