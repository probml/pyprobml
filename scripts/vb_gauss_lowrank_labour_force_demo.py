import jax.numpy as jnp
from jax.random import PRNGKey

import optax

import pandas as pd
import matplotlib.pyplot as plt

import requests
from io import BytesIO

import vb_gauss_lowrank


def logjoint_fn(w, batch, eps=0.01):
    X, y = batch
    p = 1 / (1 + jnp.exp(-jnp.dot(X, w)))
    p = jnp.clip(p, eps, 1 - eps)
    ll = jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
    lp = 0.5 * eps * jnp.sum(w ** 2)
    return -ll + lp


if __name__ == '__main__':
    # Load the data
    url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/vb_data_labour_force.csv'
    response = requests.get(url)
    rawdata = BytesIO(response.content)
    df = pd.read_csv(rawdata)
    data = df.to_numpy()

    X, y = jnp.array(data[:, :-1]), jnp.array(data[:, -1])

    learning_rate, momentum = 0.0001, 0.9
    optimizer = optax.sgd(learning_rate, momentum=momentum)

    # prior sigma for mu
    std_init = 0.01

    # Shape of mu, model params
    nfeatures = X.shape[-1]

    # initial scale
    init_scale = 0.1

    niters = 20
    window_size = 50
    num_samples = 200
    threshold = 200

    key = PRNGKey(0)
    params, lower_bounds = vb_gauss_lowrank.vb_gauss_lowrank(key, logjoint_fn, (X, y), nfeatures,
                                                             init_scale=init_scale, optimizer=optimizer,
                                                             num_samples=num_samples, niters=niters,
                                                             threshold=threshold, window_size=window_size,
                                                             smooth=False)

    print(lower_bounds)

    plt.plot(lower_bounds)
    plt.show()
