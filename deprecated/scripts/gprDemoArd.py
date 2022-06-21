import superimport

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pyprobml_utils as pml

n = 61**2
D = 2

r = np.arange(-3, 3.1, 0.1)
x1, x2 = np.meshgrid(r, r)
xx1 = x1.reshape((-1,1), order='F')
xx2 = x2.reshape((-1,1), order='F')
x = np.hstack((xx1,xx2))
Q = np.zeros((n, n))

for d in range(D):
    Q = Q + (np.tile(x[:, d].reshape((n, 1)), (1, n)) - np.tile(x[:, d].T, (n, 1)))**2
    
Q = np.exp(-0.5*Q)

np.random.seed(37)

y = scipy.linalg.cholesky(Q+1e-9*np.identity(n)).T.dot(np.random.randn(n,1))
b = y.reshape((61, 61))
fig = plt.figure(figsize=(12, 7))
ax = fig.gca(projection='3d')
ax.plot_surface(x1, x2, b, lw=0.5, cmap='viridis')
ax.set_xlabel('input x1')
ax.set_ylabel('input x2')
ax.set_zlabel('output y')

ax.view_init(None, 180+50)

plt.tight_layout()
pml.savefig('gpDemoArd0.pdf')
plt.show()


np.random.seed(0)
y = scipy.linalg.cholesky(Q+1e-9*np.identity(n)).T.dot(np.random.randn(n,1))
#%matplotlib qt
b = y.reshape((61, 61), order='F')
fig = plt.figure(figsize=(12, 7))
ax = fig.gca(projection='3d')
ax.plot_surface(x1, x2, b, lw=0.5, cmap='viridis')
ax.set_xlabel('input x1')
ax.set_ylabel('input x2')
ax.set_zlabel('output y')
ax.view_init(None, 180+50)
plt.tight_layout()
pml.savefig('gpDemoArd1.pdf')
plt.show()


###
Q = np.zeros((n, n))
L = [1, 5]
for d in range(D):
    Q = Q + (np.tile(x[:, d].reshape((n, 1)), (1, n)) - np.tile(x[:, d].T, (n, 1)))**2/L[d]**2
    
Q = np.exp(-0.5*Q)


np.random.seed(0)
y = scipy.linalg.cholesky(Q+1e-9*np.identity(n)).T.dot(np.random.randn(n,1))
b = y.reshape((61, 61), order='F')
fig = plt.figure(figsize=(12, 7))
ax = fig.gca(projection='3d')
ax.plot_surface(x1, x2, b, lw=0.5, cmap='viridis')
ax.set_xlabel('input x1')
ax.set_ylabel('input x2')
ax.set_zlabel('output y')

ax.view_init(None, 180+50)

plt.tight_layout()
pml.savefig('gpDemoArd2.pdf')
plt.show()
