#https://github.com/GPflow/GPflow/blob/master/notebooks/regression.ipynb

import GPflow
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline

# build a very simple data set:
N = 12
X = np.random.rand(N,1)
Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*0.1 + 3
plt.figure()
plt.plot(X, Y, 'kx', mew=2)

#build the GPR object
k = GPflow.kernels.Matern52(1)
meanf = GPflow.mean_functions.Linear(1,0)
m = GPflow.gpr.GPR(X, Y, k, meanf)
m.likelihood.variance = 0.01

print "Here are the parameters before optimization"
m

m.optimize()
print "Here are the parameters after optimization"
m

