import jax.numpy as jnp
from jax.random import PRNGKey
from jax import lax
from jax.scipy.stats import norm

import optax

import pandas as pd
import matplotlib.pyplot as plt

import requests
from io import BytesIO

import pyprobml_utils as pml
import vb_gauss_lowrank


def logjoint_fn(w, batch, eps=0.01):
    X, y = batch
    p = 1 / (1 + jnp.exp(-jnp.dot(X, w)))
    p = jnp.clip(p, eps, 1 - eps)
    ll = jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))
    lp = 0.5 * eps * jnp.sum(w ** 2)
    return ll - lp

def learning_rate_schedule(init_value, threshold):
    def schedule(count):
        return lax.cond(count < threshold,
                            lambda count: init_value, lambda count:init_value * threshold / count,
                            count)

    return schedule

if __name__ == '__main__':
    # Load the data
    url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/vb_data_labour_force.csv'
    response = requests.get(url)
    rawdata = BytesIO(response.content)
    df = pd.read_csv(rawdata)
    data = df.to_numpy()

    X, y = jnp.array(data[:, :-1]), jnp.array(data[:, -1])

    learning_rate, momentum = 0.001, 0.9
    learning_rate_fn = learning_rate_schedule(learning_rate, 2500)
    optimizer = optax.adafactor(learning_rate_fn, momentum=momentum)

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

    (mu, b, c), lower_bounds = vb_gauss_lowrank.vb_gauss_lowrank(key, logjoint_fn, (X, y),
                                                              nfeatures, None, std_init, init_scale,
                                                              num_samples,
                                                              niters, optimizer)



    print(lower_bounds)
    Sigma = b @ b.T + jnp.diag(c**2)


    def is_pos_def(x):
        return jnp.all(jnp.linalg.eigvals(x) > 0)
    print(is_pos_def(Sigma))

    step = 0.001

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < 8:
            x = jnp.arange(mu[i] - 3 * jnp.sqrt(Sigma[i][i]),
                           mu[i] + 3 * jnp.sqrt(Sigma[i][i]) + step, step)

            y = norm.pdf(x, mu[i], jnp.sqrt(Sigma[i][i]))
            title = f'$\Theta_{i}$'
            ax.set_title(title, fontsize=14)
            ax.plot(x, y, '-')
        else:
            ax.plot(lower_bounds)
            ax.set_title('Lower Bound')

    plt.tight_layout()
    pml.savefig("vb_gauss_lowrank_labour_force.pdf")
    pml.savefig("vb_gauss_lowrank_labour_force.png")
    plt.show()
