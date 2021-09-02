

import superimport

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = "../figures"
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from scipy.stats import t, laplace, norm
    
x = np.linspace(-4, 4, 100)
n = norm.pdf(x, loc=0, scale=1)
l = laplace.pdf(x, loc=0, scale=1 / (2 ** 0.5))
t1 = t.pdf(x, df=1, loc=0, scale=1)
t2 = t.pdf(x, df=2, loc=0, scale=1)


plt.plot(x, n, 'k:',
        x, t1, 'b--',
         x, t2, 'g--',
        x, l, 'r-')
plt.legend(('Gauss', 'Student(dof 1)', 'Student (dof 2)', 'Laplace'))
plt.ylabel('pdf')
save_fig('studentLaplacePdf2.pdf')
plt.show()

plt.figure()
plt.plot(x, np.log(n), 'k:',
        x, np.log(t1), 'b--',
        x, np.log(t2), 'g--',
        x, np.log(l), 'r-')
plt.ylabel('log pdf')
plt.legend(('Gauss', 'Student(dof 1)', 'Student (dof 2)', 'Laplace'))
#plt.legend(('Gauss', 'Student', 'Laplace'))
save_fig('studentLaplaceLogpdf2.pdf')
plt.show()
