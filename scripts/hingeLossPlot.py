import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-2, 2.01, 0.01)
L01 = (np.sign(z)).astype('float64')
Lhinge = np.maximum(np.zeros((401,)), 1-z)
Lnll = np.log2(1+np.exp(-z))
Lbinom = np.log2(1+np.exp(-2*z))
Lexp = np.exp(-z)

nllLoss = plt.plot(z, L01, 'k-')
nllLoss = plt.plot(z, Lhinge+0.02, 'b')
nllLoss = plt.plot(z, Lnll, 'r--')
nllLoss = plt.legend(['0-1','hinge','logloss'])
nllLoss = plt.rcParams.update({'legend.fontsize': 12})
nllLoss = plt.xlabel('z')
nllLoss = plt.ylabel('loss')
nllLoss = plt.show()

