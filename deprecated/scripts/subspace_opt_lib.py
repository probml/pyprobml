# Author : Kevin Murphy(@murphyk), Aleyna Kara(@karalleyna)

# import superimport

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, tree_leaves, tree_map
from jax import random
from jax.random import split, permutation
from jax.nn import one_hot
from jax.lax import scan

import optax

from functools import partial


def generate_random_basis(key, d, D):
    projection_matrix = random.normal(key, shape=(d, D))
    projection_matrix = projection_matrix / jnp.linalg.norm(projection_matrix, axis=-1, keepdims=True)
    return projection_matrix


@jit
def convert_params_from_subspace_to_full(params_subspace, projection_matrix, params_full_init):
    return jnp.matmul(params_subspace, projection_matrix)[0] + params_full_init


def data_stream(key, X, y, batch_size):
    n_data = len(X)
    while True:
        perm_key, key = split(key)
        perm = permutation(perm_key, n_data)
        num_batches, mod = divmod(n_data, batch_size)
        num_batches += 1 if mod else 0
        for i in range(num_batches):
            batch_idx = perm[i * batch_size: min((i + 1) * batch_size, n_data)]
            yield X[batch_idx], y[batch_idx]


def make_potential(key, predict_fn, dataset, batch_size, l2_regularizer):
    # Return function to compute negative log joint for each minibatch
    dataloader = data_stream(key, dataset["X"], dataset["y"], batch_size)
    n_data = dataset["X"].shape[0]

    @jit
    def loglikelihood(params, x, y):
        logits = predict_fn(params, x)
        num_classes = logits.shape[-1]
        labels = one_hot(y, num_classes)
        ll = jnp.sum(labels * logits, axis=-1)
        return ll

    @jit
    def logprior(params):
        # Spherical Gaussian prior 
        leaves_of_params = tree_leaves(params)
        return sum(tree_map(lambda p: jnp.sum(jax.scipy.stats.norm.logpdf(p, scale=l2_regularizer)), leaves_of_params))

    @jit
    def potential(params, data):
        ll = n_data * jnp.mean(loglikelihood(params, *data))
        logp = logprior(params)
        return -(ll + logp)

    @jit
    def objective(params):
        return potential(params, next(dataloader))

    return objective


def make_potential_subspace(key, anchor_params_tree, predict_fn, dataset, batch_size, l2_regularizer, subspace_dim,
                            projection_matrix=None):
    # Return function to compute negative log joint in subspace for each minibatch
    anchor_params_full, flat_to_pytree_fn = jax.flatten_util.ravel_pytree(anchor_params_tree)
    full_dim = len(anchor_params_full)

    dataloader = data_stream(key, dataset["X"], dataset["y"], batch_size)
    n_data = dataset["X"].shape[0]

    if projection_matrix is None:
        subspace_key, key = split(key)
        projection_matrix = generate_random_basis(key, subspace_dim, full_dim)

    @jit
    def subspace_to_pytree_fn(params_subspace):
        params_full = convert_params_from_subspace_to_full(params_subspace, projection_matrix, anchor_params_full)
        params_pytree = flat_to_pytree_fn(params_full)
        return params_pytree

    @jit
    def loglikelihood(params, x, y):
        logits = predict_fn(params, x)
        num_classes = logits.shape[-1]
        labels = one_hot(y, num_classes)
        ll = jnp.sum(labels * logits, axis=-1)
        return ll

    @jit
    def logprior(params):
        # Spherical Gaussian prior 
        return jnp.sum(jax.scipy.stats.norm.logpdf(params, scale=l2_regularizer))

    @jit
    def potential(params_sub, data):
        params_pytree = subspace_to_pytree_fn(params_sub)
        ll = n_data * jnp.mean(loglikelihood(params_pytree, *data))
        logp = logprior(params_sub)
        return -(ll + logp)

    @jit
    def objective(params_sub):
        return potential(params_sub, next(dataloader))

    return objective, subspace_to_pytree_fn


def optimize_loop(objective, initial_params, optimizer, n_steps, callback=None):
    opt_state = optimizer.init(initial_params)

    def train_step(carry, step):
        params, opt_state = carry
        loss, grads = value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        callback_result = callback(params, step) if callback is not None else None
        return (params, opt_state), (loss, callback_result)

    steps = jnp.arange(n_steps)
    (params, _), (loss, callback_hist) = scan(train_step, (initial_params, opt_state), steps)
    return params, loss, callback_hist
