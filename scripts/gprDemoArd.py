# Generates Figure 17.2 
# Author: Ashish Papanai (@ashishpapanai)

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

if os.path.isdir('scripts'):
    os.chdir('scripts')

np.random.seed(37)
a = 61
n = 61 ** 2
D = 2
x1, x2 = np.meshgrid(np.arange(-3, 3.1, 0.1), np.arange(-3, 3.1, 0.1))
x = (np.concatenate((x1.reshape(
    x1.shape[0]*x1.shape[1], 1), x2.reshape(x2.shape[0]*x2.shape[1], 1)), axis=1))

x[:, [0, 1]] = x[:, [1, 0]]


# Plot #1
Q = np.zeros((n, n))

for d in range(0, D):
    Q = Q + np.power(np.tile((x[:, d]).reshape(n, 1), (1, n)) -
                     (np.tile((x[:, d]).reshape(1, n), (n, 1))), 2)


Q = np.exp(np.multiply(-0.5, Q))

arr = np.transpose(Q + 1e-9 * np.eye(n))
y = np.dot(np.linalg.cholesky(arr), np.random.randn(n, 1))

fig1 = plt.figure()
ax = fig1.gca(projection='3d')
surf = ax.plot_surface(x1, x2, y.reshape((a, a)), cmap=cm.viridis)
ax.set_xlabel('input x1')
ax.set_ylabel('input x2')
ax.set_zlabel('output y')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-2, 2)
ax.grid(False)
plt.savefig('../figures/GPARD.png')
plt.show()


# Plot #2

np.random.seed(34)
Q = np.zeros((n, n))
L1 = 1
L2 = 5

Q = Q + (np.power(np.tile((x[:, 0]).reshape(n, 1), (1, n)) -
                  (np.tile((x[:, 0]).reshape(1, n), (n, 1))), 2)) / L1**2
Q = Q + (np.power(np.tile((x[:, 1]).reshape(n, 1), (1, n)) -
                  (np.tile((x[:, 1]).reshape(1, n), (n, 1))), 2)) / L2**2
Q = np.exp(np.multiply(-0.5, Q))

arr = np.transpose(Q + 1e-9 * np.eye(n))
y = np.dot(np.linalg.cholesky(arr), np.random.randn(n, 1))

fig2 = plt.figure()
ax = fig2.gca(projection='3d')
surf = ax.plot_surface(x1, x2, y.reshape((a, a)), cmap=cm.viridis)
ax.set_xlabel('input x1')
ax.set_ylabel('input x2')
ax.set_zlabel('output y')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-2, 2)
ax.grid(False)
plt.savefig('../figures/GPARD2.png')
plt.show()


# Plot #3

np.random.seed(34)
Q = np.zeros((n, n))
L1 = 1
L2 = 5

Q = Q + (np.power(np.tile((x[:, 0]).reshape(n, 1), (1, n)) -
                  (np.tile((x[:, 0]).reshape(1, n), (n, 1))), 2))

Q = Q + (np.power(np.tile((x[:, 0]).reshape(n, 1), (1, n)) -
                  (np.tile((x[:, 0]).reshape(1, n), (n, 1))), 2))/36

Q = Q + (np.power(np.tile((x[:, 1]).reshape(n, 1), (1, n)) -
                  (np.tile((x[:, 1]).reshape(1, n), (n, 1))), 2)) / 36

Q = np.exp(np.multiply(-0.5, Q))

arr = np.transpose(Q + 1e-9 * np.eye(n))
y = np.dot(np.linalg.cholesky(arr), np.random.randn(n, 1))

fig2 = plt.figure()
ax = fig2.gca(projection='3d')
surf = ax.plot_surface(x1, x2, y.reshape((a, a)), cmap=cm.viridis)
ax.set_xlabel('input x1')
ax.set_ylabel('input x2')
ax.set_zlabel('output y')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-2, 2)
ax.grid(False)
plt.savefig('../figures/GPARDFA.png')
plt.show()
