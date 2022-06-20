'''
Fits Bernoulli mixture model for mnist digits using Gradient Descent
Author: Aleyna Kara(@karalleyna)
'''

import superimport

import jax.numpy as jnp

from mix_bernoulli_lib import BMM
from mix_bernoulli_em_mnist import mnist_data

def main():
    n_obs= 1000
    observations = mnist_data(n_obs)  # subsample the MNIST dataset
    n_vars = len(observations[0])
    K, num_epochs = 12, 500
    bmm = BMM(K, n_vars)
    _ = bmm.fit_sgd(jnp.array(observations), n_obs, num_epochs=num_epochs)

    n_row, n_col = 3, 4
    bmm.plot(n_row, n_col, 'bmm_sgd_mnist')

if __name__ == "__main__":
    main()