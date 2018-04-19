# http://www.inference.vc/evolution-strategies-variational-optimisation-and-natural-es-2/
# code by arash javanmard

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Sinc function
f = lambda x: -np.sinc(x)

batch_size = 10000
N = 50
mu = np.linspace(-5, 5, N)
sigma = np.linspace(1e-1, 2, N)

rand_no = np.random.randn(1, batch_size)[0]
# Create array with different mu and sigma
rand_array = np.add.outer( np.multiply.outer(rand_no, sigma),mu)
# E = summation of all random values / total no. of samples
expected_value = np.sum(f(rand_array),0) / batch_size

# Plottig related code
MU, SIGMA = np.meshgrid(mu, sigma)


plt.figure()
plt.plot(expected_value[0,:])

plt.figure()
plt.contourf(MU, SIGMA, expected_value, cmap=cm.coolwarm)

plt.show()


    
# fig = plt.figure()
# #ax = fig.add_subplot(221, projection='3d')
# fig2, ax = plt.subplots(1, 1)
# surf = ax.plot_surface(SIGMA, MU, expected_value, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# 
# ax = fig.add_subplot(222)
# ax.plot(expected_value[0,:])
# 
# ax = fig.add_subplot(223)
# ax.contourf(MU, SIGMA, expected_value, cmap=cm.coolwarm)
# 
# plt.show()
