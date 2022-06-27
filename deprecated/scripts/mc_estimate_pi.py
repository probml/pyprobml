import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
import pyprobml_utils as pml

np.random.seed(0)
N = 5000
r = 2
xs = np.random.uniform(low=-r, high=r, size=N)
ys = np.random.uniform(low=-r, high=r, size=N)
rs = xs ** 2 + ys ** 2
inside = (rs <= r**2)
samples = 4*(r**2)*inside
Ihat = np.mean(samples)
pi_estimate = Ihat/(r**2)
se = np.sqrt(np.var(samples)/N)
print(('the estimated pi = %f' % pi_estimate))
print(('the standard pi = %f' % np.pi))
print(('stderr = %f' % se))

plt.figure(figsize=(5, 5))
plt.plot(xs[inside], ys[inside], 'bo')
plt.plot(xs[~inside], ys[~inside], 'rD')
pml.savefig('mcEstimatePi.pdf')
plt.show()
