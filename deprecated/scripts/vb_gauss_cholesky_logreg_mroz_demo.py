'''
It implements example 3.4 from https://arxiv.org/abs/2103.0132
Author: Aleyna Kara(@karalleyna)
'''

from jax import jit, random, tree_leaves, tree_map
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.random import split

import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
import optax

import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

import requests
from io import BytesIO

import pyprobml_utils as pml

# Local import
import vb_gauss_cholesky as ffvb


def make_fns_for_posterior(predict_fn, alpha):
    @jit
    def loglikelihood(params, x, y):
        predictions = predict_fn(params, x)
        ll = (y.T @ predictions - jnp.sum(predictions + jnp.log1p(jnp.exp(-predictions)))).sum()
        return ll

    @jit
    def logprior(params):
        # Spherical Gaussian prior
        return -sum(tree_leaves(tree_map(lambda x: jnp.log(alpha * x.T @ x / 2).sum(), params)))

    return loglikelihood, logprior


class LogisticRegressor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1, use_bias=False, kernel_init=nn.initializers.zeros)(x)
        return x


if __name__ == '__main__':
    key = random.PRNGKey(42)

    # Load the data
    url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/vb_data_mroz.csv'
    response = requests.get(url)
    rawdata = BytesIO(response.content)
    df = pd.read_csv(rawdata)

    data = df.to_numpy()
    X = data[:, 1:]
    y = data[:, 0]

    X = sm.add_constant(X)
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
    results = glm_binom.fit()
    mu = jnp.array(results.params)

    model = LogisticRegressor()
    init_key, key = split(key)
    variables = model.init(init_key, X)
    output = model.apply(variables, X)

    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)

    variables = unfreeze(variables)
    variables['params']['Dense_0']['kernel'] = mu.reshape((-1, 1))
    variables = freeze(variables)

    alpha = 1.
    nfeatures = tree_map(lambda x: x.shape[0], variables)
    loglikelihood_fn, logprior_fn = make_fns_for_posterior(model.apply, alpha)

    lambda_best, avg_lower_bounds = ffvb.vb_gauss_chol(key, loglikelihood_fn, logprior_fn,
                                                       (X, y), optimizer, variables,
                                                       lower_triangular=None, num_samples=20,
                                                       window_size=10, niters=150, eps=0.1)

    mu = lambda_best[0]['params']['Dense_0']['kernel']
    lower_triangular = lambda_best[1]['params']['Dense_0']['kernel']

    d_theta = 8
    *_, nfeatures = X.shape

    Sigma = lower_triangular @ lower_triangular.T
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
            ax.plot(avg_lower_bounds)
            ax.set_title('Lower Bound')

    plt.tight_layout()
    pml.savefig("vb_gauss_cholesky_mroz.pdf")
    pml.savefig("vb_gauss_cholesky_mroz.png")
    plt.show()
