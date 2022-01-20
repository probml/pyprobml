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
  inital_grad_mean = jax.tree_map(lambda n: jnp.zeros(n,), nfeatures)
  initial_grad_lower = jax.tree_map(lambda n:
                                    jnp.zeros(((n * (n+ 1))// 2,)),
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


def learning_rate_schedule(init_value, threshold):
  def schedule(count):
    if count <threshold:
      return init_value
    return (init_value * threshold) / count
  return schedule


def vechinv(v, d):
  X = jnp.zeros((d, d))
  X = ops.index_update(X, jnp.tril_indices(d, k = 0), v.squeeze())
  X = X + X.T - jnp.diag(jnp.diag(X))
  return X

def make_moving_average_fn(window_size):
  def take_average(x):
    t = len(x)
    return jnp.mean(x[t - window_size + 1 :])
  return take_average


def make_fns_for_posterior(predict_fn, nfeatures, variance=1.):
    @jit
    def loglikelihood(params, x, y):
        predictions = predict_fn(params, x)
        ll = (y.T @ predictions - jnp.sum(predictions + jnp.log1p(jnp.exp(-predictions)))).sum()
        return ll

    @jit
    def logprior(params):
        # Spherical Gaussian prior
        log_p_theta = jax.tree_multimap(lambda param, d: (-d / 2 * jnp.log(2 * jnp.pi) - d / 2 * jnp.log(variance) - (
                    param.T @ param) / 2 / variance).flatten(),
                                        params, nfeatures)

        return sum(jax.tree_leaves(log_p_theta))[0]

    return loglikelihood, logprior


def make_vb_gauss_chol_fns(variance, loglikelihood_fn, logprior_fn,
                           predict_fn, nfeatures, num_samples):
    if not loglikelihood_fn or not logprior_fn:
        loglikelihood_fn, logprior_fn = make_fns_for_posterior(predict_fn, nfeatures, variance)

    def objective(params, data):
        return -(loglikelihood_fn(params, *data) + logprior_fn(params))

    take_grad = jit(grad(objective))

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

    def step(key, variational_params, grads, data, lower_bound):
        def update(grads_and_lb, key):
            grad_mu, grad_lower, lower_bound = grads_and_lb
            params, epsilon = sample(key, variational_params)
            grad_logjoint = take_grad(params, data)
            grad_mu = jax.tree_multimap(lambda x, y: x + y.flatten(),
                                        grad_mu, grad_logjoint)
            tmp = jax.tree_multimap(jnp.outer, grad_logjoint, epsilon)
            grad_lower = jax.tree_multimap(lambda x, y: x + y[jnp.tril_indices(len(y))],
                                           grad_lower, tmp)
            lower_bound = lower_bound + objective(params, data)
            return (grad_mu, grad_lower, lower_bound), None

        keys = random.split(key, num_samples)
        (grad_mu, grad_lower, lower_bound), _ = lax.scan(update, (*grads, lower_bound), keys)

        grads = estimate_lower_bound_grad(variational_params, grad_mu, grad_lower)
        return grads, -lower_bound

    return step


def vb_gauss_chol(key, data, optimizer, mean, predict_fn, lower_triangular=None,
                  variance=10, num_samples=20, max_patience=10,
                  window_size=10, niters=500, eps=0.1,
                  loglikelihood_fn=None, logprior_fn=None):
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

    step_fn = make_vb_gauss_chol_fns(variance, loglikelihood_fn, logprior_fn,
                                     predict_fn, nfeatures, num_samples)

    take_average = make_moving_average_fn(window_size)

    t, patience, stop = 0, 0, False
    lower_bounds, avg_lower_bounds = jnp.zeros((0,)), jnp.zeros((0,))

    while not stop:
        sample_key, key = random.split(key)
        grads = init_grads(nfeatures)
        lower_bound = 0

        grads, lower_bound = step_fn(sample_key, variational_params,
                                     grads, data, lower_bound)
        grads = jax.tree_map(lambda x: x[..., None] if len(x.shape) == 1 else x, grads)
        grads = jax.tree_map(clip, grads)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        mean, std = params
        variational_params = (mean, jax.tree_multimap(lambda s, d: vechinv(s, d), std, nfeatures))
        best_params = params

        cholesky = jax.tree_map(lambda L: jnp.log(jnp.linalg.det(L * L.T)), lower_triangular)

        lb = jax.tree_multimap(
            lambda chol, n: lower_bound / num_samples + 1 / 2 * chol + n / 2,
            cholesky, nfeatures)

        lower_bounds = jnp.append(lower_bounds, sum(jax.tree_leaves(lb)))

        if t >= window_size - 1:
            avg_lower_bounds = jnp.append(avg_lower_bounds,
                                          take_average(lower_bounds))

        if t >= window_size and avg_lower_bounds[-1] >= jnp.max(avg_lower_bounds):
            best_params = params
            patience = 0
        else:
            patience = patience + 1

        if patience > max_patience or t > niters:
            stop = True

        t = t + 1

    return variational_params, avg_lower_bounds
