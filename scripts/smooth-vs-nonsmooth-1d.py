
import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

x = np.linspace(-1, 1, 100)
y = np.power(x, 2)
plt.figure()
plt.plot(x, y, '-', lw=3)
plt.title('Smooth function')
pml.savefig('smooth-fn.pdf')

y = np.abs(x)
plt.figure()
plt.plot(x, y, '-', lw=3)
plt.title('Non-smooth function')
pml.savefig('nonsmooth-fn.pdf')

plt.show()