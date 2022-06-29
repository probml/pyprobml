'''
Fits Bernoulli mixture model for mnist digits using em algorithm
Author: Meduri Venkata Shivaditya, Aleyna Kara(@karalleyna)
'''

import superimport

from jax.random import PRNGKey, randint
import tensorflow as tf
from mix_bernoulli_lib import BMM

def mnist_data(n_obs, rng_key=None):
    '''
    Downloads data from tensorflow datasets
    Parameters
    ----------
    n_obs : int
        Number of digits randomly chosen from mnist
    rng_key : array
        Random key of shape (2,) and dtype uint32
    Returns
    -------
    * array((n_obs, 784))
        Dataset
    '''
    rng_key = PRNGKey(0) if rng_key is None else rng_key

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x = (x_train > 0).astype('int')  # Converting to binary
    dataset_size = x.shape[0]

    perm = randint(rng_key, minval=0, maxval=dataset_size, shape=((n_obs,)))
    x_train = x[perm]
    x_train = x_train.reshape((n_obs, 784))

    return x_train

def main():
    n_obs= 1000
    observations = mnist_data(n_obs)  # subsample the MNIST dataset
    n_vars = len(observations[0])
    K, num_of_iters = 12, 10
    n_row, n_col = 3, 4
    bmm = BMM(K, n_vars)
    _ = bmm.fit_em(observations, num_of_iters=num_of_iters)
    bmm.plot(n_row, n_col, 'bmm_em_mnist')

if __name__ == "__main__":
    main()