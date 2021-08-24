# Plot theoretical rates of convergence

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

plt.figure(figsize=(12,4))

ks = range(1,10)
ys = [1.0/k for k in ks]
print(ys)
plt.subplot(1,3,1)
plt.plot(ks, np.log(ys), color = 'r')
plt.title('Sublinear convergence')

ys = [1.0/(2**k) for k in ks]
print(ys)
plt.subplot(1,3,2)
plt.plot(ks, np.log(ys), color = 'g')
plt.title('Linear convergence')

ys = [1.0/(2**(2**k)) for k in ks]
print(ys)
plt.subplot(1,3,3)
plt.plot(ks, np.log(ys), color = 'b')
plt.title('Quadratic convergence')

#fig.subplots_adjust(hspace=0)
plt.tight_layout()
plt.draw()

pml.savefig('convergenceRates.pdf')
plt.show()
