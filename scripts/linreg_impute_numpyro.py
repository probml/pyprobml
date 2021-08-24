import superimport

import numpy as np

import jax.numpy as jnp
from jax import ops

import numpyro
import numpyro.distributions as dist


def linreg_model(X, y):

    ndims = X.shape[1]
    a = numpyro.sample("a", dist.Normal(0, 0.5))

    beta = numpyro.sample("beta", dist.Normal(0, 0.5).expand([ndims]))
    sigma_y = numpyro.sample("sigma_y", dist.Exponential(1))

    # LKJ is the distribution to model correlation matrices.
    rho = numpyro.sample("rho", dist.LKJ(ndims, 2))  # correlation matrix
    sigma_x = numpyro.sample("sigma_x", dist.Exponential(1).expand([ndims]))
    covariance_x = jnp.outer(sigma_x, sigma_x) * rho  # covariance matrix
    mu_x = numpyro.sample("mu_x", dist.Normal(0, 0.5).expand([ndims]))

    numpyro.sample("X", dist.MultivariateNormal(mu_x, covariance_x), obs=X)

    mu_y = a + X @ beta

    numpyro.sample("y", dist.Normal(mu_y, sigma_y), obs=y)


def linreg_imputation_model(X, y):
  
    ndims = X.shape[1]
    a = numpyro.sample("a", dist.Normal(0, 0.5))

    beta = numpyro.sample("beta", dist.Normal(0, 0.5).expand([ndims]))
    sigma_y = numpyro.sample("sigma_y", dist.Exponential(1))

    # X_impute contains imputed data for each feature as a list
    # X_merged is the observed data filled with imputed values at missing points.
    X_impute = [None] * ndims
    X_merged = [None] * ndims

    for i in range(ndims):  # for every feature
        no_of_missed = int(np.isnan(X[:, i]).sum())

        if no_of_missed != 0:
            # each nan value is associated with a imputed variable of std normal prior.
            X_impute[i] = numpyro.sample(
                "X_impute_{}".format(i), dist.Normal(0, 1).expand([no_of_missed]).mask(False))

            # merging the observed data with the imputed values.
            missed_idx = np.nonzero(np.isnan(X[:, i]))[0]
            X_merged[i] = ops.index_update(X[:, i], missed_idx, X_impute[i])

        # if there are no missing values, its just the observed data.
        else:
            X_merged[i] = X[:, i]

    merged_X = jnp.stack(X_merged).T

    # LKJ is the distribution to model correlation matrices.
    rho = numpyro.sample("rho", dist.LKJ(ndims, 2))  # correlation matrix
    sigma_x = numpyro.sample("sigma_x", dist.Exponential(1).expand([ndims]))
    covariance_x = jnp.outer(sigma_x, sigma_x) * rho  # covariance matrix
    mu_x = numpyro.sample("mu_x", dist.Normal(0, 0.5).expand([ndims]))

    numpyro.sample("X_merged", dist.MultivariateNormal(mu_x, covariance_x), obs=merged_X)

    mu_y = a + merged_X @ beta

    numpyro.sample("y", dist.Normal(mu_y, sigma_y), obs=y)
