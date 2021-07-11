

import numpy as np
import matplotlib.pyplot as plt
import os

def save_fig(fname):
    figdir = os.path.join(os.environ["PYPROBML"], "figures")
    plt.tight_layout()    
    fullname = os.path.join(figdir, fname)
    print('saving to {}'.format(fullname))
    plt.savefig(fullname)
   
from scipy.stats import gaussian_kde
from scipy.stats import norm
np.random.seed(42)

x = np.arange(0.5, 2.5, 0.01)
for size in [10, 100, 1000]:
    samples = norm.rvs(loc=1.5, scale=0.5, size=size)
    y = norm.pdf(x, loc=1.5, scale=0.5)

    plt.figure()
    #plt.hist(samples, normed=True, rwidth=0.8)
    plt.hist(samples, density=True, rwidth=0.8)
    plt.plot(x, y, 'r')
    plt.xlim(0, 3)
    plt.title('n_samples = %d' % size)
    save_fig('mcAccuracyDemoHist%d.pdf' % size)

    kde = gaussian_kde(samples)
    y_estimate = kde(x)
    plt.figure()
    plt.plot(x, y, 'r', label='true pdf')
    plt.plot(x, y_estimate, 'b--', label='estimated pdf')
    plt.legend()
    plt.title('n_samples = %d' % size)
    save_fig('mcAccuracyDemoKde%d.pdf' % size)
plt.show()
