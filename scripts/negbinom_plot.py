

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml


from scipy.stats import nbinom



xs = np.arange(50);
fig, ax = plt.subplots(1,1)
p = 0.5; r = 1;
probabilities = nbinom.pmf(xs, r, p)
ax.bar(xs, probabilities)
ax.set_title('NB(r={:.2f}, p={:.2f})'.format(r, p))
pml.savefig('negbinomPlot_r{}_p{}.pdf'.format(int(r*10), int(p*10)))
plt.show()

fig, ax = plt.subplots(1,1)
p = 0.5; r = 10;
probabilities = nbinom.pmf(xs, r, p)
ax.bar(xs, probabilities)
ax.set_title('NB(r={:.2f}, p={:.2f})'.format(r, p))
pml.savefig('negbinomPlot_r{}_p{}.pdf'.format(int(r*10), int(p*10)))
plt.show()


