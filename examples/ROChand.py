#!/usr/bin/env python

# ROC curves for two hypothetical classification systems
# A is better than B. Plots true positive rate, (tpr) vs false positive
# rate, (fpr).

import matplotlib.pyplot as pl
import numpy as np

fA = np.vectorize(lambda x: x**(1.0/3))
fB = np.vectorize(lambda x: x**(2.0/3))
x = np.arange(0, 1, 0.01)

pl.plot(x, fA(x), 'r-')
pl.plot(x, fB(x), 'b-')
pl.fill_between(x, fB(x), 0, facecolor='blue')
pl.plot(x, 1-x, 'k-')

inter_a = 0.3177; # found using scipy.optimize.fsolve(x**(1.0/3)+x-1, 0)
inter_b = 0.4302; # found using scipy.optimize.fsolve(x**(2.0/3)+x-1, 0)

pl.plot(inter_a, fA(inter_a), 'ro')
pl.plot(inter_b, fB(inter_b), 'bo')

pl.text(inter_a, fA(inter_a) + 0.1, 'A', color='red', size='x-large')
pl.text(inter_b, fB(inter_b) + 0.1, 'B', color='blue', size='x-large')

pl.xlabel('FPR', size='xx-large')
pl.ylabel('TPR', size='xx-large')
pl.savefig('ROChand.png')
pl.show()
