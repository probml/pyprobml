#!/usr/bin/env python

# Plots the sigmoid function.

import matplotlib.pyplot as plt
import numpy as np
import os

e = np.exp(1)
x = np.linspace(-10, 10, 1000)
y = e**x / (e**x + 1)
plt.plot(x, y)
plt.title('sigmoid function')
plt.show()
plt.savefig(os.path.join('../figures', 'sigmoidPlot.pdf'))

