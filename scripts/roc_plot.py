
# ROC curves for two hypothetical classification systems
# A is better than B. Plots true positive rate, (tpr) vs false positive
# rate, (fpr).

import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml



fA = np.vectorize(lambda x: x**(1.0/3))
fB = np.vectorize(lambda x: x**(2.0/3))
x = np.arange(0, 1, 0.01)

plt.plot(x, fA(x), 'r-')
plt.plot(x, fB(x), 'b-')
plt.fill_between(x, fB(x), 0, color='gray', alpha=0.2)
plt.plot(x, 1-x, 'k-')

inter_a = 0.3177 # found using scipy.optimize.fsolve(x**(1.0/3)+x-1, 0)
inter_b = 0.4302 # found using scipy.optimize.fsolve(x**(2.0/3)+x-1, 0)

plt.plot(inter_a, fA(inter_a), 'ro')
plt.plot(inter_b, fB(inter_b), 'bo')

plt.text(inter_a, fA(inter_a) + 0.1, 'A', color='red', size='x-large')
plt.text(inter_b, fB(inter_b) + 0.1, 'B', color='blue', size='x-large')

plt.xlabel('FPR', fontsize=14)
plt.ylabel('TPR', fontsize=14)
pml.savefig('ROChand.pdf')
plt.show()
