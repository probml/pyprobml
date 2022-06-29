import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

from scipy.stats import beta


x = np.linspace(0, 1, 100)
aa = [0.1, 0.1, 1.0, 2.0, 2.0]
bb = [0.1, 1.0, 1.0, 2.0, 8.0]
#props = ['b-', 'r:', 'k-.', 'g--', 'c-']
props = ['b', 'r', 'k', 'g', 'c']
for a, b, p in zip(aa, bb, props):
    y = beta.pdf(x, a, b)
    plt.plot(x, y, p, lw=3, label='a=%.1f,b=%.1f' % (a, b))
plt.legend(fontsize=14)
plt.title('Beta distributions')
pml.savefig('betadist.pdf', dpi=300)
plt.show()
