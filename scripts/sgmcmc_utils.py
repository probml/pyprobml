# Various extensions to https://github.com/jeremiecoullon/SGMCMCJax
# Author : Kevin Murphy(@murphyk), Aleyna Kara(@karalleyna)

import jax.numpy as jnp
from jax import jit, lax, random, tree_map, vmap
from jax.random import split
from jax.flatten_util import ravel_pytree

from blackjax import nuts, stan_warmup
import optax
from sgmcmcjax.util import build_grad_log_post, progress_bar_scan
from sgmcmcjax.gradient_estimation import build_gradient_estimation_fn


# Extends https://github.com/jeremiecoullon/SGMCMCJax/blob/master/sgmcmcjax/optimizer.py
def build_optax_optimizer(optimizer, loglikelihood, logprior, data, batch_size, pbar: bool = True):
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
        return (key, state, params), (lp_val, ravel_pytree(params)[0])

    def run_optimizer(key, Niters, params):
        state = optimizer.init(params)
        lebody = progress_bar_scan(Niters)(body) if pbar else body
        (key, state, params), (logpost_array, params_trace) = lax.scan(lebody, (key, state, params), jnp.arange(Niters))
        return params, logpost_array, params_trace

    return run_optimizer


# Extends https://github.com/jeremiecoullon/SGMCMCJax/blob/master/sgmcmcjax/samplers.py
# by making a wrapper for blackjax.nuts (https://github.com/blackjax-devs/blackjax),
# so it acts like other sgmcmc samplers (ie takes loglikelihood and logprior, instead of potential)

def inference_loop(rng_key, kernel, initial_state, num_samples, pbar):
    def one_step(carry, i):
        state, key = carry
        kernel_key, key = split(key)
        state, _ = kernel(kernel_key, state)
        return (state, key), state

    lebody = progress_bar_scan(num_samples)(one_step) if pbar else one_step
    # _, states = lax.scan(one_step, (initial_state, rng_key), jnp.arange(num_samples))
    _, states = lax.scan(lebody, (initial_state, rng_key), jnp.arange(num_samples))
    return states


def build_log_post(loglikelihood, logprior, data):
    if len(data) == 1:
        batch_loglik = jit(vmap(loglikelihood, in_axes=(None, 0)))
    elif len(data) == 2:
        batch_loglik = jit(vmap(loglikelihood, in_axes=(None, 0, 0)))
    else:
        raise ValueError("Data must be a tuple of size 1 or 2")

    def log_post(params):
        return logprior(params) + jnp.sum(batch_loglik(params, *data), axis=0)

    return jit(log_post)


def build_nuts_sampler(nwarmup, loglikelihood, logprior, data, batch_size=None, pbar=True):
    # wrapper for blackjax, so it acts like other sgmcmc samplers
    log_post = build_log_post(loglikelihood, logprior, data)
    ndata = data.shape[0]

    def potential(params):
        v = log_post(params) / ndata  # scale down by N to avoid numerical problems
        return -v

    def nuts_sampler(rng_key, num_samples, initial_params):
        initial_state = nuts.new_state(initial_params, potential)

        kernel_generator = lambda step_size, inverse_mass_matrix: jit(nuts.kernel(
            potential, step_size, inverse_mass_matrix))
        stan_key, key = split(rng_key)

        final_state, (step_size, inverse_mass_matrix), _ = stan_warmup.run(
            stan_key,
            kernel_generator,
            initial_state,
            nwarmup)

        nuts_kernel = kernel_generator(step_size, inverse_mass_matrix)

        inference_key, key = split(key)
        states = inference_loop(inference_key, nuts_kernel, final_state, num_samples, pbar)
        return states.position

    return nuts_sampler
