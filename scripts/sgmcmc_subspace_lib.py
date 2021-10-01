# Author : Kevin Murphy(@murphyk)

import jax
import jax.numpy as jnp
from jax import jit, tree_leaves, lax, random, tree_map, vmap
from jax.random import split

import optax

from blackjax import nuts, stan_warmup

from sgmcmcjax.util import build_grad_log_post, progress_bar_scan
from sgmcmcjax.gradient_estimation import build_gradient_estimation_fn


def generate_random_basis(key, d, D):
    projection_matrix = random.normal(key, shape=(d, D))
    projection_matrix = projection_matrix / jnp.linalg.norm(projection_matrix, axis=-1, keepdims=True)
    return projection_matrix


@jit
def convert_params_from_subspace_to_full(params_subspace, projection_matrix, params_full_init):
    return jnp.matmul(params_subspace, projection_matrix)[0] + params_full_init


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
        params_pytree = subspace_to_pytree_fn(params_subspace)
        return logprior(params_pytree)

    return loglikelihood_subspace, logprior_subspace, subspace_to_pytree_fn


def subspace_sampler(key, loglikelihood, logprior, params_init_tree, build_sampler, data, batch_size, subspace_dim,
                     nsteps_full, nsteps_sub, nsamples, use_cv=True, opt=optax.adam(learning_rate=0.1)):
    # Find good control variate / starting point in subspace
    opt_key, sample_key = split(key)
    params_tree, params_sub, log_post_trace, loglik_sub, logprior_sub, subspace_to_pytree_fn = subspace_optimizer(
        opt_key, loglikelihood, logprior, params_init_tree, data, batch_size, subspace_dim, nsteps_full, nsteps_sub,
        opt)

    if use_cv:
        sampler_sub = build_sampler(loglikelihood=loglik_sub, logprior=logprior_sub, data=data, batch_size=batch_size,
                                    centering_value=params_sub)
    else:
        sampler_sub = build_sampler(loglikelihood=loglik_sub, logprior=logprior_sub, data=data, batch_size=batch_size)

    params_sub_samples = sampler_sub(sample_key, nsamples, params_sub)
    params_tree_samples = vmap(subspace_to_pytree_fn)(params_sub_samples)

    return params_tree_samples


def subspace_optimizer(key, loglikelihood, logprior, params_init_tree, data, batch_size, subspace_dim, nwarmup, nsteps,
                       opt=optax.adam(learning_rate=0.1), projection_matrix=None):
    optimizer = build_optax_optimizer(opt, loglikelihood, logprior, data, batch_size)

    opt_key, subspace_key, sub_init_key, sub_opt_key = split(key, 4)

    # Find good anchor in full space during warmup phase
    anchor_params_tree, _ = optimizer(opt_key, nwarmup, params_init_tree)
    full_dim = len(tree_leaves(anchor_params_tree))

    # Make Random subspace
    if projection_matrix is None:
        projection_matrix = generate_random_basis(subspace_key, subspace_dim, full_dim)

    loglik_sub, logprior_sub, subspace_to_pytree_fn = make_subspace_fns(loglikelihood, logprior, anchor_params_tree,
                                                                        projection_matrix)

    # Do subspace optimization starting from rnd location
    params_subspace = jax.random.normal(sub_init_key, (subspace_dim,))
    optimizer_sub = build_optax_optimizer(opt, loglik_sub, logprior_sub, data, batch_size)

    params_subspace, log_post_trace = optimizer_sub(sub_opt_key, nsteps, params_subspace)
    params_tree = subspace_to_pytree_fn(params_subspace)

    return params_tree, params_subspace, log_post_trace, loglik_sub, logprior_sub, subspace_to_pytree_fn


def build_optax_optimizer(optimizer, loglikelihood, logprior, data, batch_size):
    grad_log_post = build_grad_log_post(loglikelihood, logprior, data, with_val=True)
    estimate_gradient, _ = build_gradient_estimation_fn(grad_log_post, data, batch_size)

    @jit
    def body(carry, i):
        key, state, params = carry
        key, subkey = random.split(key)
        (lp_val, param_grad), _ = estimate_gradient(i, subkey, params)
        neg_param_grad = tree_map(lambda x: -x, param_grad)
        updates, state = optimizer.update(neg_param_grad, state)
        params = optax.apply_updates(params, updates)
        return (key, state, params), lp_val

    def run_optimizer(key, Niters, params):
        state = optimizer.init(params)
        body_pbar = progress_bar_scan(Niters)(body)
        (key, state, params), logpost_array = lax.scan(body_pbar, (key, state, params), jnp.arange(Niters))
        return params, logpost_array

    return run_optimizer


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @progress_bar_scan(num_samples)
    def one_step(carry, i):
        state, key = carry
        kernel_key, key = split(key)
        state, _ = kernel(kernel_key, state)
        return (state, key), state

    _, states = lax.scan(one_step, (initial_state, rng_key), jnp.arange(num_samples))
    return states


def build_nuts_sampler(num_warmup, potential):
    # wrapper for blackjax, so it acts like other sgmcmc samplers
    def nuts_sampler(rng_key, num_samples, initial_params):
        initial_state = nuts.new_state(initial_params, potential)

        kernel_generator = lambda step_size, inverse_mass_matrix: jit(nuts.kernel(
            potential, step_size, inverse_mass_matrix))

        stan_key, key = split(rng_key)

        final_state, (step_size, inverse_mass_matrix), _ = stan_warmup.run(
            kernel_generator,
            initial_state,
            num_warmup)

        nuts_kernel = kernel_generator(step_size, inverse_mass_matrix)

        inference_key, key = split(key)
        states = inference_loop(inference_key, nuts_kernel, final_state, num_samples)
        return states.position

    return nuts_sampler
