# Create a version of fig 4.19 from "Hands-on ML with Scikit-Learn" 
# by Aurelien Geron
# Based on his original code here:
# https://nbviewer.jupyter.org/github/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os

# plot range for theta1 and theta2
t1a, t1b, t2a, t2b = -1, 3, -1.5, 1.5

# create a grid of parameter combinations
t1s = np.linspace(t1a, t1b, 500)
t2s = np.linspace(t2a, t2b, 500)
t1, t2 = np.meshgrid(t1s, t2s)
params = np.c_[t1.ravel(), t2.ravel()]

# create some data
np.random.seed(1)
N = 10
D = 2
X = np.random.randn(N, D)
y = np.c_[2 * X[:, 0] + 0.5 * X[:, 1]]

# compute MSE for all possible parameters
J = (1/N * np.sum((params.dot(X.T) - y.T)**2, axis=1)).reshape(t1.shape)

# Compute norm of parameter values
N1 = np.linalg.norm(params, ord=1, axis=1).reshape(t1.shape)
N2 = np.linalg.norm(params, ord=2, axis=1).reshape(t1.shape)


# Initial value for gradient descent
t_init = np.array([[0.25], [-1]])


# batch gradient descent
def bgd_path(theta, X, y, l1, l2, include_mse = 1, eta = 0.1, n_iterations = 50):
    path = [theta]
    for iteration in range(n_iterations):
        gradients = include_mse * 2/len(X) * X.T.dot(X.dot(theta) - y) + \
            l1 * np.sign(theta) + 2 * l2 * theta
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)


def do_plot(l1=0, l2=0, ttl="", figdir=None):
    JR = J + l1*N1 + l2*N2**2
    N = l1*N1 + l2*N2**2
    tr_min_idx = np.unravel_index(np.argmin(JR), JR.shape)
    t1r_min, t2r_min = t1[tr_min_idx], t2[tr_min_idx]

    levelsJR=(np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(JR) - np.min(JR)) + np.min(JR)
    levelsN=np.linspace(0, np.max(N), 10)
    path_JR = bgd_path(t_init, X, y, l1=l1, l2=l2)

    plt.figure(figsize=(6, 4))
    plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.contourf(t1, t2, JR, levels=levelsJR, alpha=0.9)
    if l1>0 or l2>0:
        plt.contour(t1, t2, N, levels=levelsN)
    plt.plot(path_JR[:, 0], path_JR[:, 1], "w-o")
    plt.plot(t1r_min, t2r_min, "rs")
    plt.axis([t1a, t1b, t2a, t2b])
    plt.xlabel(r"$\theta_1$", fontsize=20)
    plt.ylabel(r"$\theta_2$", fontsize=20, rotation=0)
    plt.title(ttl, fontsize=16)
    if not(figdir is None):
      fname = os.path.join(figdir, "{}.png".format(ttl))
      plt.savefig(fname)
    plt.show()


figdir = "../figures"
do_plot(l1=0, l2=0, ttl="OLS-2d", figdir=figdir)
do_plot(l1=0.5, l2=0, ttl="lasso-2d", figdir=figdir)
do_plot(l1=0, l2=0.1, ttl="ridge-2d", figdir=figdir)

