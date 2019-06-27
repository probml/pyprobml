# Precision-recall curve for two hypothetical classifications systems.
# A is better than B. 

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


fA = np.vectorize(lambda x: 1 - x**3)
fB = np.vectorize(lambda x: 1 - x**(3/2))
x = np.arange(0, 1, 0.01)
plt.plot(x, fA(x), 'r-')
plt.plot(x, fB(x), 'b-')

plt.text(0.6, 0.8, 'A', color='red', size='x-large')
plt.text(0.1, 0.8, 'B', color='blue', size='x-large')

plt.axis([0, 1, 0, 1.01])
plt.xlabel('recall', size='xx-large')
plt.ylabel('precision', size='xx-large')
plt.legend()
save_fig('PRhand.pdf')
plt.show()
