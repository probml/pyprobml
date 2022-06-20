import superimport

import pyprobml_utils as pml
import numpy as np
from scipy.stats import uniform, laplace, norm
import matplotlib.pyplot as plt

n = 2000
x = np.arange(-4, 4, 0.01)
y1 = norm.pdf(x, 0, 1)
y2 = uniform.pdf(x, -2, 4)
y3 = laplace.pdf(x, 0, 1)

plt.plot(x, y1, color='blue')
plt.plot(x, y2, color='green')
plt.plot(x, y3, color='red')
pml.savefig('1D.pdf')
plt.savefig('1D.pdf')
plt.show()

x1 = np.random.normal(0, 1, n).reshape(n, 1)
x2 = np.random.normal(0, 1, n).reshape(n, 1)
plt.scatter(x1, x2, marker='.', color='blue')
plt.gca().set_aspect('equal')
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.title("Gaussian")
pml.savefig('Gaussian.pdf')
plt.savefig('Gaussian.pdf')
plt.show()

x1 = np.random.laplace(0, 1, n).reshape(n, 1)
x2 = np.random.laplace(0, 1, n).reshape(n, 1)
plt.scatter(x1, x2, marker='.', color='red')
plt.gca().set_aspect('equal')
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.title("Laplace")
pml.savefig('Laplace.pdf')
plt.savefig('Laplace.pdf')
plt.show()

x1 = np.random.uniform(-2, 2, n).reshape(n, 1)
x2 = np.random.uniform(-2, 2, n).reshape(n, 1)
plt.scatter(x1, x2, marker='.', color='green')
plt.gca().set_aspect('equal')
plt.xlim(-2.5, 2.5)
plt.ylim(-2, 2)
plt.title("Uniform")
pml.savefig('Uniform.pdf')
plt.savefig('Uniform.pdf')
plt.show()
