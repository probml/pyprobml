# Conjugate function Illustration
# This code is based on https://github.com/probml/pmtk3/blob/master/demos/conjugateFunction.m

import superimport

import numpy as np
import matplotlib.pyplot as plt
import math
import pyprobml_utils as pml

start, stop, step = 0.01, 5, 0.01
domain = np.arange(start, stop + step, step)
offset = 0.04

f = lambda x: -math.log(5 * x) - 5 if isinstance(x, int) else -np.log(5 * x) - 5
t = lambda x: -(1 / 3) * x + f(3) + 1 - offset
t1 = lambda x: t(x) - 0.5

# figureA
for i in range(1, 5):
    plt.plot([i, i], [t1(i), f(i)], '--g')
plt.plot(domain, t1(domain), '-b', linewidth=3)
plt.plot(domain, f(domain), '-r', linewidth=5)
plt.xlim([0, 5])
plt.ylim([-9, -6])
plt.text(1, t1(1) - 1 / 2, r'$\lambda$x', {'fontsize': 20})
plt.text(1, f(1) - 1 / 2, 'f(x)', {'fontsize': 20})
pml.savefig('conjugate_functionA.pdf', dpi=300)
plt.show()

# figureB
plt.plot(domain, t(domain), '-b', linewidth=3)
plt.plot(domain, f(domain), '-r', linewidth=5)
plt.xlim([0, 5])
plt.ylim([-9, -6])
plt.yticks([t(0)], ['-f*($\lambda$)'], fontsize=12)
plt.text(3, t(3) - 1 / 2, r'$\lambda$x = f*($\lambda$)', {'fontsize': 15})
plt.text(1, f(1) + 1 / 3, 'f(x)', {'fontsize': 15})
pml.savefig('conjugate_functionB.pdf', dpi=300)
plt.show()
