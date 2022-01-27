'''
It implements the full covariance FFVB method from 3.5.1 of https://arxiv.org/abs/2103.01327
For original Matlab code, please see Example4.zip in https://github.com/VBayesLab/Tutorial-on-VB.
Author : Aleyna Kara(@karalleyna)
'''

import jax
import jax.numpy as jnp
from jax import random, jit, grad, lax, ops

import optax


def init_grads(nfeatures):
    inital_grad_mean = jax.tree_map(lambda n: jnp.zeros(n, ), nfeatures)
    initial_grad_lower = jax.tree_map(lambda n:
                                      jnp.zeros(((n * (n + 1)) // 2,)),
                                      nfeatures)
    return inital_grad_mean, initial_grad_lower


def clip(X, threshold=10, norm=None):
    # gradient clipping
    if norm is None:
        X_leaves = jax.tree_leaves(X)
        norm = sum(jax.tree_map(jnp.linalg.norm, X_leaves))

    def true_fun(x):
        return (threshold / norm) * x

    def false_fun(x):
        return x

    X = jax.tree_map(lambda x: jax.lax.cond(norm > threshold, true_fun, false_fun, x), X)
    return X


def vechinv(v, d):
    X = jnp.zeros((d, d))
    X = ops.index_update(X, jnp.tril_indices(d, k=0), v.squeeze())
    return X


def make_vb_gauss_chol_fns(loglikelihood_fn, logprior_fn, nfeatures, num_samples):
    def logjoint(params, data):
        return -loglikelihood_fn(params, *data) - logprior_fn(params)

    take_grad = jit(grad(logjoint))

    def sample(key, variational_params):
        # Take a single sample from a Gaussian distribution.
        mean, std = variational_params
        epsilon = jax.tree_map(lambda x: random.normal(key, x.shape), mean)

        params = jax.tree_multimap(lambda mu, sigma, eps: mu + sigma @ eps,
                                   mean, std, epsilon)
        return params, epsilon

    def estimate_lower_bound_grad(variational_params, grad_mu, grad_lower):
        grad_mu = jax.tree_map(lambda x: x / num_samples, grad_mu)

        _, std = variational_params
        diagonal = jax.tree_map(lambda L: jnp.diag(jnp.diag(L)), std)
        grad_lower = jax.tree_multimap(lambda dL, D, n: dL / num_samples + D[jnp.tril_indices(n)],
                                       grad_lower, diagonal, nfeatures)
        return grad_mu, grad_lower

    def step(key, variational_params, grads, data):
        def update(grads_and_lb, key):
            grad_mu, grad_lower, lower_bound = grads_and_lb
            params, epsilon = sample(key, variational_params)
            grad_logjoint = take_grad(params, data)
            grad_mu = jax.tree_multimap(lambda x, y: x + y.flatten(),
                                        grad_mu, grad_logjoint)
            tmp = jax.tree_multimap(jnp.outer, grad_logjoint, epsilon)
            grad_lower = jax.tree_multimap(lambda x, y: x + y[jnp.tril_indices(len(y))],
                                           grad_lower, tmp)
            lower_bound = lower_bound + logjoint(params, data)
            return (grad_mu, grad_lower, lower_bound), None

        keys = random.split(key, num_samples)
        lower_bound = 0

        (grad_mu, grad_lower, lower_bound), _ = lax.scan(update, (*grads, lower_bound), keys)
        grads = estimate_lower_bound_grad(variational_params, grad_mu, grad_lower)

        return grads, -lower_bound

    return step


def vb_gauss_chol(key, loglikelihood_fn, logprior_fn,
                  data, optimizer, mean,
                  lower_triangular=None, num_samples=20,
                  window_size=10, niters=500,
                  eps=0.1, smooth=True):
    '''
    Arguments:
      num_samples : number of Monte Carlo samples,
      mean : prior mean of the distribution family

    '''

    nfeatures = jax.tree_map(lambda x: x.shape[0], mean)

    if lower_triangular is None:
        # initializes the lower triangular matrices
        lower_triangular = jax.tree_map(lambda n: eps * jnp.eye(n), nfeatures)

    # Initialize parameters of the model + optimizer.
    variational_params = (mean, lower_triangular)

    params = (mean, jax.tree_multimap(lambda L, n: L[jnp.tril_indices(n)][..., None],
                                      lower_triangular, nfeatures))

    opt_state = optimizer.init(params)

    step_fn = make_vb_gauss_chol_fns(loglikelihood_fn, logprior_fn,
                                     nfeatures, num_samples)

    def iter_fn(all_params, key):
        variational_params, params, opt_state = all_params
        grads = init_grads(nfeatures)

        lower_bound = 0
        grads, lower_bound = step_fn(key, variational_params, grads, data)

        grads = jax.tree_map(lambda x: x[..., None] if len(x.shape) == 1 else x, grads)
        grads = clip(grads)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        mean, std = params
        variational_params = (mean, jax.tree_multimap(lambda s, d: vechinv(s, d), std, nfeatures))
        cholesky = jax.tree_map(lambda L: jnp.log(jnp.linalg.det(L @ L.T)), variational_params[1])

        lb = jax.tree_multimap(lambda chol, n: lower_bound / num_samples + 1 / 2 * chol + n / 2,
                               cholesky, nfeatures)

        return (variational_params, params, opt_state), (variational_params, lb)

    keys = jax.random.split(key, niters)
    _, (variational_params, lower_bounds) = jax.lax.scan(iter_fn, (variational_params, params, opt_state), keys)
    lower_bounds = jax.tree_leaves(lower_bounds)[0]

    if smooth:
        def simple_moving_average(cur_sum, i):
            diff = (lower_bounds[i] - lower_bounds[i - window_size]) / window_size
            cur_sum += diff
            return cur_sum, cur_sum

        indices = jnp.arange(window_size, niters)
        cur_sum = jnp.sum(lower_bounds[:window_size]) / window_size
        _, lower_bounds = jax.lax.scan(simple_moving_average, cur_sum, indices)
        lower_bounds = jnp.append(jnp.array([cur_sum]), lower_bounds)

    i = jnp.argmax(lower_bounds) + window_size - 1 if smooth else jnp.argmax(lower_bounds)
    best_params = jax.tree_map(lambda x: x[i], variational_params)

    return best_params, lower_bounds
