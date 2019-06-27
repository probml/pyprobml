

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from scipy.stats import t, laplace, norm
    
x = np.linspace(-4, 4, 100)
n = norm.pdf(x, loc=0, scale=1)
l = laplace.pdf(x, loc=0, scale=1 / (2 ** 0.5))
t = t.pdf(x, df=1, loc=0, scale=1)


plt.plot(x, n, 'k:',
        x, t, 'b--',
        x, l, 'r-')
plt.legend(('Gauss', 'Student', 'Laplace'))
plt.ylabel('pdf')
save_fig('studentLaplacePdf.pdf')
plt.show()

plt.figure()
plt.plot(x, np.log(n), 'k:',
        x, np.log(t), 'b--',
        x, np.log(l), 'r-')
plt.ylabel('log pdf')
plt.legend(('Gauss', 'Student', 'Laplace'))
save_fig('studentLaplaceLogpdf.pdf')
plt.show()
