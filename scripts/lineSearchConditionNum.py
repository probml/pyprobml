
import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, line_search


import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad, value_and_grad, hessian, jacfwd, jacrev

# Objective is a quadratic
# f(x) = 0.5 x'Ax + b'x + c


def gradient_descent(x0, f, f_prime, hessian, stepsize = None, nsteps=50):
    """
                    Steepest-Descent algorithm with option for line search
    """
    x_i, y_i = x0
    all_x_i = list()
    all_y_i = list()
    all_f_i = list()

    for i in range(1, nsteps):
        all_x_i.append(x_i)
        all_y_i.append(y_i)
        x = np.array([x_i, y_i])
        all_f_i.append(f(x))
        dx_i, dy_i = f_prime(x)
        if stepsize is None:
            # Compute a step size using a line_search to satisfy the Wolf
            # conditions
            step = line_search(f, f_prime,
                                np.r_[x_i, y_i], -np.r_[dx_i, dy_i],
                                np.r_[dx_i, dy_i], c2=.05)
            step = step[0]
            if step is None:
                step = 0
        else:
            step = stepsize
        x_i += - step*dx_i
        y_i += - step*dy_i
        if np.abs(all_f_i[-1]) < 1e-5:
            break
    return all_x_i, all_y_i, all_f_i


def make_plot(A, b, c, fname):
    def objective(x): # x is (2,)
        f =  jnp.dot(x, jnp.dot(A, x)) + jnp.dot(x,b) + c
        return f
    
    
    def objective_vectorized(X): # x is (N,2)
        f = vmap(objective)(X)
        return f
    
    
    def gradient(x):
        return jnp.dot(A + A.T, x) + b
    
    def hessian(x):
        return A
    
    z = objective_vectorized(X)
    N = len(x1)
    z = np.reshape(z, (N,N))
    plt.contour(x1, x2, z, 50)
    x0 = np.array((0.0, 0.0))
    #x0 = np.array((-1.0, -1.0))
    xs, ys, fs = gradient_descent(x0, objective, gradient, hessian, stepsize = None)
    nsteps = 20
    plt.scatter(xs[:nsteps], ys[:nsteps])
    plt.plot(xs[:nsteps], ys[:nsteps])
    plt.title('condition number of A={:0.3f}'.format(np.linalg.cond(A)))
    plt.tight_layout()
    plt.savefig('../figures/{}.pdf'.format(fname), dpi=300)
    plt.show()



x1 = np.arange(-2, 2, 0.1) # xs is (40,)
x2 = x1
xs, ys = np.meshgrid(x1, x2) # xs is (1600,)
X=np.stack((xs.flatten(), ys.flatten()),axis=1) # (1600,2)

A = np.array([[20,5],[5,2]])
b = np.array([-14,-6])
c = 10
fname = 'steepestDescentCondNumBig'
make_plot(A, b, c, fname)

A = np.array([[20,5],[5,16]]) 
fname = 'steepestDescentCondNumSmall'
make_plot(A, b, c, fname)

