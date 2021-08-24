

# Plots loss functions of form |x|**q
import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from scipy.stats import t, laplace, norm

x = np.linspace(-4, 4, 100)
plt.title('|x|^0.2')
plt.plot(x, np.absolute(x)**.2)
pml.savefig('lossFnQ2.pdf')

plt.figure()
plt.title('|x|')
plt.plot(x, np.absolute(x))
pml.savefig('lossFnQ10.pdf')

plt.figure()
plt.title('|x|^2')
plt.plot(x, np.absolute(x)**2)
pml.savefig('lossFnQ20.pdf')
plt.show()
