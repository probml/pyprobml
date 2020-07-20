

import numpy as np
import matplotlib.pyplot as plt
import os
#figdir = os.path.join(os.environ["PYPROBML"], "figures")
figdir = "../figures";
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))


from scipy.stats import nbinom



xs = np.arange(50);
fig, ax = plt.subplots(1,1)
p = 0.5; r = 1;
probabilities = nbinom.pmf(xs, r, p)
ax.bar(xs, probabilities)
ax.set_title('NB(r={:.2f}, p={:.2f})'.format(r, p))
save_fig('negbinomPlot_r{}_p{}.png'.format(int(r*10), int(p*10)))
plt.show()

fig, ax = plt.subplots(1,1)
p = 0.5; r = 10;
probabilities = nbinom.pmf(xs, r, p)
ax.bar(xs, probabilities)
ax.set_title('NB(r={:.2f}, p={:.2f})'.format(r, p))
save_fig('negbinomPlot_r{}_p{}.png'.format(int(r*10), int(p*10)))
plt.show()


