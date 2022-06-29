
import superimport

import numpy as np
import matplotlib.pyplot as plt

ts = np.arange(100)
tis = np.array([25, 50, 75])
lr_list = []
lr0 = 1
gamma = 0.9
for t in ts:
    passed_num_thresholds = sum(t > tis)
    lr = lr0 * np.power(gamma, passed_num_thresholds)
    lr_list.append(lr)
    
plt.figure()
plt.plot(lr_list)
plt.title('piecewise constant')
pml.savefig('lr_piecewise_constant.pdf')
plt.show()


ts = np.arange(100)
lam = 0.999
lr0 = 1
lr_list = lr0 * np.exp(-lam*ts)

plt.figure()
plt.plot(lr_list)
plt.title('exponential decay')
plt.savefig('lr_exp_decay.pdf')
plt.show()

ts = np.arange(100)
alpha = 0.5
beta = 1
lr0 = 1
lr_list = lr0 * np.power(beta*ts + 1, -alpha)

plt.figure()
plt.plot(lr_list)
plt.title('polynomial decay')
pml.savefig('lr_poly_decay.pdf')
plt.show()
