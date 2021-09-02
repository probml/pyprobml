# Precision-recall curve for two hypothetical classifications systems.
# A is better than B. 

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml


fA = np.vectorize(lambda x: 1 - x**3)
fB = np.vectorize(lambda x: 1 - x**(3/2))
x = np.arange(0, 1, 0.01)
plt.plot(x, fA(x), 'r-', linewidth=3)
plt.plot(x, fB(x), 'b-', linewidth=3)

plt.text(0.6, 0.8, 'A', color='red', size='xx-large')
plt.text(0.1, 0.8, 'B', color='blue', size='xx-large')

plt.axis([0, 1, 0, 1.01])
plt.xlabel('recall', fontsize=14)
plt.ylabel('precision', fontsize=14)
plt.legend()
pml.savefig('PRhand.pdf')
plt.show()
