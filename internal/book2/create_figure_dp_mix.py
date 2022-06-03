# Sample from a DP mixture of 2D Gaussians
# Converted from 
# https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book2/30/dpm_sample_demo.ipynb#scrollTo=86e42eea

import numpy as np
import jax.numpy as jnp
from jax import random

from NIW import NormalInverseWishart


def dp_mixture_simu(N, alpha, H, key):
    """
    Generating samples from the Gaussian Dirichlet process mixture model.
    We set the base measure of the DP to be Normal Inverse Wishart (NIW)
    and the likelihood be multivariate normal distribution 
    ------------------------------------------------------
    N: int 
        Number of samples to be generated from the mixture model
    alpha: float
        Concentration parameter of the Dirichlet process 
    H: object of NormalInverseWishart
        Base measure of the Dirichlet process
    key: jax.random.PRNGKey
        Seed of initial random cluster
    --------------------------------------------
    * array(N):
        Simulation of cluster assignment
    * array(N, dimension):
        Simulation of samples from the DP mixture model
    * array(K, dimension):
        Simulation of mean of each cluster
    * array(K, dimension, dimension):
        Simulation of covariance of each cluster
    """
    Z = jnp.full(N, 0)
    # Sample cluster assignment from the Chinese restaurant process prior 
    CR = []
    for i in range(N):
        p = jnp.array(CR + [alpha])
        key, subkey = random.split(key)
        k = random.categorical(subkey, logits=jnp.log(p))
        # Add new cluster to the mixture 
        if k == len(CR):
            CR = CR + [1]
        # Increase the size of corresponding cluster by 1 
        else:
            CR[k] += 1
        Z = Z.at[i].set(k)
    # Sample the parameters for each component of the mixture distribution, from the base measure 
    key, subkey = random.split(key)
    params = H.sample(seed=subkey, sample_shape=(len(CR),))
    Sigma = params['Sigma'] 
    Mu = params['mu']
    # Sample from the mixture distribtuion
    subkeys = random.split(key, N)
    X = [random.multivariate_normal(subkeys[i], Mu[Z[i]], Sigma[Z[i]]) for i in range(N)]
    return Z, jnp.array(X), Mu, Sigma


if __name__=="__main__":
    
    from jax.scipy.linalg import sqrtm
    import matplotlib.pyplot as plt

    
    # Example
    dim = 2
    # Set the hyperparameter for the NIW distribution
    hyper_params = dict(
        loc = jnp.zeros(dim),
        mean_precision = 0.05,
        df = dim + 5,
        scale = jnp.eye(dim)
    )
    # Generate the NIW object
    niw = NormalInverseWishart(**hyper_params)

    # Plot
    N = 1000
    alpha = [1.0, 2.0]

    bb = np.arange(0, 2 * np.pi, 0.02)
    ss = [50, 500, 1000]
    fig, axes = plt.subplots(3, 2)
    plt.setp(axes, xticks=[], yticks=[])
    
    for i in range(2):
        Z, X, Mu, Sigma = dp_mixture_simu(N, alpha[i], niw, random.PRNGKey(3))
        Sig_root = jnp.array([sqrtm(sigma) for sigma in Sigma])
        for j in range(3):
            s = ss[j]
            axes[j,i].plot(X[:s, 0], X[:s, 1], ".", markersize=5)
            for k in jnp.unique(Z[:s]):
                sig_root = Sig_root[k, ]
                mu = Mu[[k],].T
                circ = mu.dot(np.ones((1, len(bb)))) + sig_root.dot(np.vstack([np.sin(bb), np.cos(bb)]))
                axes[j,i].plot(circ[0, :], circ[1, :], linewidth=2, color="k")
    plt.show()
            