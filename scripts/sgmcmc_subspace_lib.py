# Author : Kevin Murphy(@murphyk), Aleyna Kara(@karalleyna)

import jax
import jax.numpy as jnp
from jax import jit, random, vmap
from jax.random import split

import optax

from sgmcmc_utils import build_optax_optimizer


def generate_random_basis(key, d, D):
    projection_matrix = random.normal(key, shape=(d, D))
    projection_matrix = projection_matrix / jnp.linalg.norm(projection_matrix, axis=-1, keepdims=True)
    return projection_matrix


@jit
def convert_params_from_subspace_to_full(params_subspace, projection_matrix, params_full_init):
    return jnp.matmul(params_subspace, projection_matrix) + params_full_init


def make_subspace_fns(loglikelihood, logprior, anchor_params_tree, projection_matrix):
    anchor_params_full, flat_to_pytree_fn = jax.flatten_util.ravel_pytree(anchor_params_tree)

    def subspace_to_pytree_fn(params_subspace):
        params_full = convert_params_from_subspace_to_full(params_subspace, projection_matrix, anchor_params_full)
        params_pytree = flat_to_pytree_fn(params_full)
        return params_pytree

    def loglikelihood_subspace(params_subspace, *args):
        params_pytree = subspace_to_pytree_fn(params_subspace)
        return loglikelihood(params_pytree, *args)

    def logprior_subspace(params_subspace):
        # return 0
        params_pytree = subspace_to_pytree_fn(params_subspace)
        return logprior(params_pytree)

    return loglikelihood_subspace, logprior_subspace, subspace_to_pytree_fn


def subspace_optimizer(key, loglikelihood, logprior, anchor_params_tree, data, batch_size, subspace_dim, nsteps,
                       opt=optax.adam(learning_rate=0.1), nwarmup=None, projection_matrix=None, pbar=True):
    optimizer = build_optax_optimizer(opt, loglikelihood, logprior, data, batch_size, pbar)

    opt_key, subspace_key, sub_init_key, sub_opt_key = split(key, 4)

    # Find good anchor in full space during warmup phase
    if nwarmup is not None:
        anchor_params_tree, _ = optimizer(opt_key, nwarmup, anchor_params_tree)

    anchor_params_full, _ = jax.flatten_util.ravel_pytree(anchor_params_tree)
    full_dim = len(anchor_params_full)

    # Make Random subspace
    if projection_matrix is None:
        projection_matrix = generate_random_basis(subspace_key, subspace_dim, full_dim)

    loglik_sub, logprior_sub, subspace_to_pytree_fn = make_subspace_fns(loglikelihood, logprior, anchor_params_tree,
                                                                        projection_matrix)

    # Do subspace optimization starting from rnd location
    params_subspace = jax.random.normal(sub_init_key, (subspace_dim,))
    optimizer_sub = build_optax_optimizer(opt, loglik_sub, logprior_sub, data, batch_size, pbar)

    params_subspace, log_post_trace = optimizer_sub(sub_opt_key, nsteps, params_subspace)
    params_tree = subspace_to_pytree_fn(params_subspace)

    subspace_fns = (loglik_sub, logprior_sub, subspace_to_pytree_fn)
    return params_tree, params_subspace, log_post_trace, subspace_fns


def subspace_sampler(key, loglikelihood, logprior, params_init_tree, build_sampler, data, batch_size,
                     subspace_dim, nsteps_full, nsteps_sub, nsamples, use_cv=True,
                     opt=optax.adam(learning_rate=0.1), pbar=True):
    # Find good control variate / starting point in subspace
    opt_key, sample_key = split(key)
    params_tree, params_sub, log_post_trace, subspace_fns = subspace_optimizer(
        opt_key, loglikelihood, logprior, params_init_tree, data, batch_size,
        subspace_dim, nsteps_sub, opt, nwarmup=nsteps_full, pbar=pbar)
    loglik_sub, logprior_sub, subspace_to_pytree_fn = subspace_fns
    if use_cv:
        sampler_sub = build_sampler(loglikelihood=loglik_sub, logprior=logprior_sub, data=data, batch_size=batch_size,
                                    centering_value=params_sub, pbar=pbar)
    else:
        sampler_sub = build_sampler(loglikelihood=loglik_sub, logprior=logprior_sub, data=data,
                                    batch_size=batch_size, pbar=pbar)

    params_sub_samples = sampler_sub(sample_key, nsamples, params_sub)
    params_tree_samples = vmap(subspace_to_pytree_fn)(params_sub_samples)

    return params_tree_samples
