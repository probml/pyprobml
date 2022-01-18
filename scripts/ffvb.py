import jax
import jax.numpy as jnp
from jax import random, jit, grad, lax, ops
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import unfreeze, freeze

import optax


def init_grads(nfeatures):
    inital_grad_mean = jax.tree_map(lambda n: jnp.zeros((n,1)), nfeatures)
    initial_grad_lower = jax.tree_map(lambda n:
                                      jnp.zeros(((n * (n + 1)) // 2,)),
                                      nfeatures)
    return inital_grad_mean, initial_grad_lower


def clip(X, threshold=10, norm=None):
    # gradient clipping
    def true_fun(x):
        return (threshold / norm) * x

    def false_fun(x):
        return x

    X = jax.tree_map(lambda x: jax.lax.cond(norm > threshold, true_fun, false_fun, x), X)
    return X


def learning_rate_schedule(init_value, threshold):
    def schedule(count):
        return jax.lax.cond(count < threshold, 
                            lambda count: init_value, lambda count:init_value * threshold / count,
                            count)

    return schedule


def vechinv(v, d):
    X = jnp.zeros((d, d))
    X = ops.index_update(X, jnp.tril_indices(d, k=0), v)
    X = X + X.T - jnp.diag(jnp.diag(X))
    return X

def make_moving_average_fn(window_size):
    def take_average(x, t):
        return jnp.mean(jax.lax.dynamic_slice(x, (t - window_size + 1,),
                                              (window_size,)))
    return take_average


def make_fns_for_posterior(predict_fn, nfeatures, variance=1.):
    @jit
    def loglikelihood(params, x, y):
        predictions = predict_fn(params, x)
        ll = (y.T @ predictions - jnp.sum(predictions + jnp.log1p(jnp.exp(-predictions)))).sum()
        return ll

    @jit
    def logprior(params, std):
        # Spherical Gaussian prior
        log_p_theta = jax.tree_multimap(lambda param, d: (-d / 2 * jnp.log(2 * jnp.pi) - d / 2 * jnp.log(variance) - (
                param.T @ param) / 2 / variance).flatten(),
                                        params, nfeatures)

        return sum(jax.tree_leaves(log_p_theta))[0]

    return loglikelihood, logprior


def make_vb_gauss_chol_fns(variational_params, variance, loglikelihood_fn,
                           logprior_fn, predict_fn, num_samples):
    mean, std = variational_params
    
    nfeatures = jax.tree_map(lambda x: x.size, mean)
    
    if not loglikelihood_fn or not logprior_fn:
        loglikelihood_fn, logprior_fn = make_fns_for_posterior(predict_fn, nfeatures, variance)

    def logjoint(params, data):
        return loglikelihood_fn(params, *data) + logprior_fn(params, std)

    take_grad = jit(grad(logjoint))
    
    
    def sample(key):
        # Take a single sample from a Gaussian distribution.
        epsilon = jax.tree_map(lambda x: random.normal(key, (x.size,)), mean)
        params = jax.tree_multimap(lambda mu, sigma, eps: mu + (sigma @ eps).reshape((mu.shape)),
                                   mean, std, epsilon)
        return params, epsilon 
    
    def estimate_lower_bound_grad(grad_mu, grad_lower):
        grad_mu = freeze(unflatten_dict(jax.tree_map(lambda x: x / num_samples, grad_mu)))
        diagonal = jax.tree_map(lambda L: jnp.diag(jnp.diag(L)), std)
        grad_lower = jax.tree_multimap(lambda dL, D, n: dL / num_samples + D[jnp.tril_indices(n)],
                                       freeze(unflatten_dict(grad_lower)), diagonal, nfeatures)
        return grad_mu, grad_lower
    
    def step(key, grads, data, lower_bound):
        def update(grads_and_lb, key):
            grad_mu, grad_lower, lower_bound = grads_and_lb
            params, epsilon = sample(key)
            
            grad_logjoint = take_grad(params, data)
            
            grad_logjoint = flatten_dict(unfreeze(grad_logjoint))
            epsilon = flatten_dict(unfreeze(epsilon))
            grad_mu = {k: v + grad_logjoint[k].reshape(v.shape) if k[-1] == 'kernel' else v
                       for k, v in grad_mu.items()}

            tmp = {k: jnp.outer(v.flatten(), epsilon[k]) if k[-1] == 'kernel' else v
                       for k, v in grad_logjoint.items()}
            grad_lower = {k: grad_lower[k] + v[jnp.tril_indices(len(v))] if k[-1] == 'kernel' else grad_lower[k]
                       for k, v in tmp.items()}

            lower_bound = lower_bound + logjoint(params, data)
            return (grad_mu, grad_lower, lower_bound), None

        keys = random.split(key, num_samples)
        
        grad_mu, grad_lower = grads
        grad_mu = flatten_dict(unfreeze(grad_mu))
        grad_lower = flatten_dict(unfreeze(grad_lower))
        (grad_mu, grad_lower, lower_bound), _ = lax.scan(update,
                                                (grad_mu, grad_lower, lower_bound), keys)

        lower_bound_grads = estimate_lower_bound_grad(grad_mu, grad_lower)
        return lower_bound_grads, lower_bound

    return step


def vb_gauss_chol(key, data, optimizer, prior_mean, predict_fn, lower_triangular=None,
                  prior_variance=10, num_samples=20, max_patience=10,
                  window_size=10, niters=500, eps=0.1,
                  loglikelihood_fn=None, logprior_fn=None):
    '''
    Arguments:
      num_samples : number of Monte Carlo samples
    '''
    '''    nfeatures = {k : v.shape[0] for k, v in flatten_dict(unfreeze(prior_mean)).items()}
        nfeatures = freeze(unflatten_dict(nfeatures))'''
    nfeatures = jax.tree_map(lambda x: x.size, prior_mean)

    if lower_triangular is None:
        lower_triangular = jax.tree_map(lambda n: eps * jnp.eye(n), nfeatures)

    # Initialize parameters of the model + optimizer.
    variational_params = (prior_mean, lower_triangular)
    params = (prior_mean, jax.tree_multimap(lambda L, n: L[jnp.tril_indices(n)],
                                            lower_triangular, nfeatures))
    opt_state = optimizer.init(params)

    step_fn = make_vb_gauss_chol_fns(variational_params, prior_variance, loglikelihood_fn,
                                     logprior_fn, predict_fn, num_samples)

    t, patience, stop = 0, 0, False
    grads = init_grads(nfeatures)

    def iterate(input, key):
        opt_state, params = input
        lower_bound = 0

        (grad_mu, grad_lower), lower_bound = step_fn(key, grads, data, lower_bound)
        
        grad_lb = jax.tree_multimap(jnp.append, grad_mu, grad_lower)
        norms = jax.tree_map(jnp.linalg.norm, grad_lb)

        grad_mu = jax.tree_multimap(lambda x, norm, mu: clip(x, norm=norm).reshape((mu.shape)),
                                    grad_mu, norms, prior_mean)
        grad_lower = jax.tree_multimap(lambda x, norm: clip(x, norm=norm),
                                    grad_lower, norms)
        grad_lb = (grad_mu, grad_lower)
        updates, opt_state = optimizer.update(grad_lb, opt_state)
        
        params = optax.apply_updates(params, updates)
        _, lower_triangular = params
        
        low_mat = jax.tree_map(vechinv, lower_triangular, nfeatures)
        cholesky = jax.tree_map(lambda L: jnp.log(jnp.linalg.det(L @ L.T)), low_mat)

        lower_bound = jax.tree_multimap(
            lambda chol, n: lower_bound / num_samples + 1 / 2 * chol + n / 2,
            cholesky, nfeatures)

        lower_bound = sum(jax.tree_leaves(lower_bound))
        return (opt_state, params), (params, lower_bound)

    
    keys = jax.random.split(key, niters)
    (_, last_params), (all_params, lower_bounds) = jax.lax.scan(iterate, (opt_state, params), keys)
    
    take_average = make_moving_average_fn(window_size)
    
    def select(prev, t):
      prev_lb, old_t = prev
      cur_lb = take_average(lower_bounds, t)
      
      max_lb = jnp.where(cur_lb >= prev_lb, cur_lb, prev_lb)
      max_t = jnp.where(cur_lb >= prev_lb, t, old_t)
      return (max_lb, max_t), cur_lb
    
    (_, t), avg_lower_bounds = jax.lax.scan(select, (lower_bounds[0], 0), jnp.arange(window_size, niters - window_size))
    
    return jax.tree_map(lambda x:x[t], all_params), avg_lower_bounds
