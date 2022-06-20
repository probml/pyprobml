# Lowerbound illustration
# This file is based on https://github.com/probml/pmtk3/blob/master/demos/optLowerbound.m

import superimport

import numpy as np
import math
import matplotlib.pyplot as plt
import pyprobml_utils as pml

start, stop, step = 0, 3, 0.01
domain = np.arange(start, stop + step, step)
offset = 0.015

f = lambda x: math.exp(-x) if isinstance(x, int) else np.exp(-x)
t = lambda x, y: -math.exp(-y) * x + f(y) + math.exp(-y) * y - offset

t1 = lambda x: t(x, 1)
t2 = lambda x: t(x, 0.5)
t3 = lambda x: t(x, 2)

plt.plot(domain, t1(domain), '-b', linewidth=3)
plt.plot(domain, t2(domain), '-g', linewidth=3)
plt.plot(domain, t3(domain), '-g', linewidth=3)
plt.plot([1, 1], [0, f(1)], '-k', linewidth=4)
plt.plot(domain, f(domain), '-r', linewidth=5)
plt.xlim([start, stop])
plt.ylim([0, 1])
plt.text(1 - 1 / 15, 0 - 1 / 15, r'$\epsilon$', {'fontsize': 20})
plt.xticks([0, 1.5, 3])
plt.yticks([0, 0.5, 1])
pml.savefig('opt_lower_bound.pdf', dpi=300)
plt.show()
