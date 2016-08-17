#https://github.com/pymanopt/pymanopt/blob/master/pymanopt/core/problem.py

import autograd.numpy as np
from pymanopt import Problem


def cost(theta):
    return np.square(theta)
    
problem = Problem(manifold=None, cost=cost, verbosity=1)

print problem.cost(5)

print problem.egrad(5.0)