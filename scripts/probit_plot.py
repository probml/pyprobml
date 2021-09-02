# Plots Sigmoid vs. Probit.

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

from scipy.special import expit
from scipy.stats import norm

x = np.arange(-6, 6, 0.1)
l = np.sqrt(np.pi/8); 
plt.plot(x, expit(x), 'r-', label='sigmoid')
plt.plot(x, norm.cdf(l*x), 'b--', label='probit')

plt.axis([-6, 6, 0, 1])
plt.legend()
pml.savefig('probitSigmoidPlot.pdf')
plt.show()
