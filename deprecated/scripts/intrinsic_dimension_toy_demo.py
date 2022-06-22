# This demo replicates Figure 1 (right) of the paper
# "Measuring the intrinsic dimension of objective landscapes" (2018) by
# Chunyuan Li and Heerad Farkhoor and Rosanne Liu and Jason Yosinski
# https://arxiv.org/pdf/1804.08838.pdf

# Author: Gerardo Durán-Martín (@gerdm)

import superimport

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from jax import random
from functools import partial
from jax.scipy.optimize import minimize


def subspace_loss(theta_sub, P, theta_0, y):
    """    
    Objective low-dimensional function takes a vector theta_sub
    such that dim(theta_sub) < dim(theta_0). We project theta_sub
    onto dim(theta_0), and reshape the input vector into a matrix
    of the form (10, ?). The loss is given by summing over each
    row of the matrix and computing the squared error between the
    resulting vector and the true y.

    Parameters
    ----------
    theta_sub : ndarray
        Low-dimensional vector of size dim(theta_sub) <  dim(theta_0)
    P : ndarray
        Projection matrix of size dim(theta_0) x dim(theta_sub)
    theta_0 : ndarray
        Full-dimensional vector
    y : ndarray
        True y
    
    Returns
    -------
    float: Loss
    """
    theta = P @ theta_sub + theta_0
    y_hat = theta.reshape(10, -1).sum(axis=1)
    return jnp.sum((y_hat - y) ** 2)


def full_dimension_loss(theta, y):
    """
    Objective full-dimensional function.
    The loss is given by summing over each
    row of the matrix and computing the squared error between the
    resulting vector and the true y.

    Parameters
    ----------
    theta : ndarray
        Full-dimensional vector
    y : ndarray
        True y
    """
    y_hat = theta.reshape(10, -1).sum(axis=1)
    return jnp.sum((y_hat - y) ** 2)

def optimize_subspace(key, d, D):
    """
    Optimize the subspace loss function for a given dimension d.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key
    d : int
        Dimension of the subspace
    
    Returns
    -------
    jax._src.scipy.optimize.minimize.OptimizeResults: Optimization results
    """
    key_weight, key_map, key_sign = random.split(key, 3)
    theta_0 = random.normal(key_weight, (D,)) / 10
    theta_sub_0 = jnp.zeros(d)
    
    choice_map = random.bernoulli(key_map, 1 / jnp.sqrt(D), shape=(D, d))
    P = random.choice(key_sign, jnp.array([-1, 1]), shape=(D, d)) * choice_map
    f_part = partial(subspace_loss, P=P, theta_0=theta_0, y=y)
    res = minimize(f_part, theta_sub_0, method="bfgs", tol=1e-3)
    return res

if __name__ == "__main__":
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    D = 1000
    R = 10
    y = jnp.arange(R) + 1

    # 1. Obtain optimal loss for the full-dimensional function
    theta_0 = jnp.zeros(D)
    f_part = partial(full_dimension_loss, y=y)
    res = minimize(f_part, theta_0, method="bfgs")
    optimal_loss = res.fun

    # 2. Obtain optimal loss for the subspace function at
    #    different dimensions
    dimensions = jnp.array(list(range(1, 16)) + [20, 30, 30])
    key = random.PRNGKey(314)
    keys = random.split(key, len(dimensions))

    ans = {
        "dim": [],
        "loss":[],
        "w": []
    }

    for key, dim in zip(keys, dimensions):
        print(f"@dim={dim}", end="\r")
        res = optimize_subspace(key, dim, D)
        ans["dim"].append(dim)
        ans["loss"].append(res.fun)
        ans["w"].append(res.x)
    
    # 3. Plot performance of the subspace function
    #   as a function of the dimension
    performance = jnp.exp(-jnp.array(ans["loss"])) / jnp.exp(-optimal_loss)
    plt.plot(dimensions, performance, marker="o")
    plt.xlabel("Subspace dim $d$")
    plt.ylabel("Performance")
    pml.savefig("intrinsic-dimension-toy-demo.pdf")
    plt.show()
