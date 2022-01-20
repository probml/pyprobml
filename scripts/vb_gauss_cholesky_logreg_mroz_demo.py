'''

!pip install -q statsmodels
'''

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
import optax

import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

import requests
from io import BytesIO

# Local import
import vb_gauss_cholesky as ffvb


class LinearRegressor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1, use_bias=False)(x)
        return x


if __name__ == '__main__':
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

    model = LinearRegressor()
    variables = model.init(jax.random.PRNGKey(0), X)
    output = model.apply(variables, X)

    start_learning_rate, tau_threshold = 1e-3, 100
    b1, b2 = 0.6, 0.6

    learning_rate_fn = ffvb.learning_rate_schedule(start_learning_rate, tau_threshold)
    optimizer = optax.adam(learning_rate_fn, b1=b1, b2=b2)

    variables = unfreeze(variables)
    variables['params']['Dense_0']['kernel'] = mu.reshape((-1, 1))
    variables = freeze(variables)

    (lambda_best, lower_triangular), avg_lower_bounds = ffvb.vb_gauss_chol(jax.random.PRNGKey(42), (X, y), optimizer,
                                                        variables, model.apply)

    lambda_best = lambda_best['params']['Dense_0']['kernel']
    lower_triangular = ffvb.vechinv(lower_triangular['params']['Dense_0']['kernel'],8)
    *_, nfeatures = X.shape

    mu = lambda_best[:nfeatures]
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
    plt.show()
