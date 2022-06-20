# Upper bounds for sigmoid function

import superimport

import numpy as np
import math
import matplotlib.pyplot as plt
import pyprobml_utils as pml

sigmoid = lambda x: np.exp(x) / (1 + np.exp(x))
fstar = lambda eta: -eta * math.log(eta) - (1 - eta) * math.log(1 - eta)
sigmoid_upper = lambda eta, x: np.exp(eta * x - fstar(eta))

eta1, eta2 = 0.2, 0.7
start, stop, step = -6, 6, 1 / 10
xs = np.arange(start, stop + step, step)


plt.plot(xs, sigmoid(xs), 'r', linewidth=3)
plt.plot(xs, sigmoid_upper(eta1, xs), 'b', linewidth=3)
plt.plot(xs, sigmoid_upper(eta2, xs), 'b', linewidth=3)
plt.text(1 / 2 + 1 / 2, sigmoid_upper(eta1, 1 / 2), 'eta=0.2')
plt.text(0 + 1 / 2, sigmoid_upper(eta2, 0), 'eta=0.7')
plt.xlim([start, stop])
plt.ylim([0, 1])
pml.savefig('sigmoid_upper_bound.pdf', dpi=300)
plt.show()
