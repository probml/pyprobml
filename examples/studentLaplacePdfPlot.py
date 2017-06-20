#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
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
plt.savefig('figures/studentLaplacePdfPlot_1.pdf')

plt.figure()
plt.plot(x, np.log(n), 'k:',
        x, np.log(t), 'b--',
        x, np.log(l), 'r-')
plt.ylabel('log pdf')
plt.legend(('Gauss', 'Student', 'Laplace'))
plt.savefig('figures/studentLaplacePdfPlot_2.pdf')

plt.show()
