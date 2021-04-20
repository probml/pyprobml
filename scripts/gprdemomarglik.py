import pyprobml_utils as pml

#!pip install GPy

import numpy as np
import matplotlib.pyplot as plt 
import GPy
from numpy.linalg import cholesky, det, inv, lstsq
from scipy.linalg import solve_triangular
import scipy.spatial.distance as spdist
from scipy import linalg
from scipy.optimize import minimize

# Dataset according to matlab code
n = 7
np.random.seed(20) 
xs = 15*(np.random.uniform(low=0, high=1, size=n) - 0.5).reshape((-1, 1))
sigma_y = 0.1
def K(p, q):
  p = p.transpose()
  p = p.transpose()
  p = np.tile(p, len(q))
  q = np.tile(q, len(p))
  r = 0.5*np.square(p-q)
  return np.exp(r)
w = K(xs, xs) + (sigma_y**2)*np.eye(n)
w = w.conjugate()
fs = np.linalg.cholesky(w).dot(np.random.randn(n, 1)).reshape((-1, 1))

#xs = np.array([-1.3089, 6.7612, 1.0553, -1.1734, -2.9339, 7.2530, -6.5843]).reshape(-1, 1)
#fs = np.array([1.6218, 1.8558, 0.4102, 1.2526, -0.0133, 1.6380, 0.2189]).reshape(-1, 1)

n=41
x = np.linspace(0.1, 150, n)
y = np.linspace(0.03, 3, n)
X, Y = np.meshgrid(x, y)

def kernel(X1, X2, l, sigma_f):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def mll(params):
        theta = params[0]
        noise = params[1]
        K = kernel(X_train, X_train, l=theta, sigma_f=1) + noise**2 * np.eye(len(X_train))
        L = cholesky(K)
        return np.sum(np.log(np.diagonal(L))) + 0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) + 0.5 * len(X_train) * np.log(2*np.pi)

X = X.flatten().reshape((-1, 1))
Y = Y.flatten().reshape((-1, 1))
Z = np.empty((n*n, 1))
X_train = xs
Y_train = fs
params = np.empty((2, ))
for i in range(n*n):
  params[0] = X[i]
  params[1] = Y[i]
  Z[i] = mll(params)

resu = minimize(mll, x0=[1, 0.1], method='L-BFGS-B')

l_opt = resu.x[0]
sigma_y_opt = resu.x[1]

Z = Z.reshape((n, n))
X = X.reshape((n, n))
Y = Y.reshape((n, n))
level = -1*np.array([15, 11.5, 9.8,  9.3,  8.9, 8.5, 8.3])

plt.plot(l_opt, sigma_y_opt, 'o')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Length-scale")
plt.ylabel("Noise-level")
plt.title("Log-marginal-likelihood")
plt.contour(X, Y, -Z) #, levels = level)
plt.title("Log-marginal likelihood")
pml.save_fig('Contour.png')
plt.savefig('Contour.png')
plt.show()

l = np.array([1.0, 10])
sigma_f = np.array([1, 1])
sigma_y = np.array([0.2, 0.8])
xstar = np.linspace(-7.5, 7.5, 201)
xstar = xstar.reshape((-1, 1))

def generate_plots(sigma_f, l, sigma_y):
    kernel = GPy.kern.RBF(1, sigma_f, l) 
    model = GPy.models.GPRegression(xs , fs, kernel) 
    model.Gaussian_noise.variance = sigma_y**2
    model.Gaussian_noise.variance.fix()
    mean, variance = model.predict(xstar)
    model.plot()
    plt.title("Hyperparameters (l, sigma_f, sigma_y) are {}, {}, {}".format(l, sigma_f, sigma_y))
    pml.save_fig('Plot' + '-' + str(i) + '.png')
    plt.savefig('Plot' + '-' + str(i) + '.png')
    plt.show()

for i in range(len(l)):
    generate_plots(sigma_f[i], l[i], sigma_y[i])
