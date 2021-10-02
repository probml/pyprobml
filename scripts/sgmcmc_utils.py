# Various extensions to https://github.com/jeremiecoullon/SGMCMCJax
# Author : Kevin Murphy(@murphyk), Aleyna Kara(@karalleyna)

import jax.numpy as jnp
from jax import jit, lax, random, tree_map, vmap
from jax.random import split

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
        return (key, state, params), lp_val

    def run_optimizer(key, Niters, params):
        state = optimizer.init(params)
        lebody = progress_bar_scan(Niters)(body) if pbar else body
        (key, state, params), logpost_array = lax.scan(lebody, (key, state, params), jnp.arange(Niters))
        return params, logpost_array

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


def build_nuts_sampler(num_warmup, loglikelihood, logprior, data, batchsize=None, pbar=True):
    # wrapper for blackjax, so it acts like other sgmcmc samplers
    log_post = build_log_post(loglikelihood, logprior, data)

    def potential(params):
        v = log_post(params)
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
            num_warmup)

        nuts_kernel = kernel_generator(step_size, inverse_mass_matrix)

        inference_key, key = split(key)
        states = inference_loop(inference_key, nuts_kernel, final_state, num_samples, pbar)
        return states.position

    return nuts_sampler

from typing import (Any, Callable, Iterable, Optional, Tuple, Union)

from flax.linen.module import Module, compact
from flax.linen.initializers import lecun_normal, zeros, normal

default_kernel_init = lecun_normal()
PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


class ProjectedDense(Module):
    features: int
    subspace_dim: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    subspace_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = normal
    subspace_bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = normal
    projection_matrix_init: Callable[[PRNGKey, Shape, Dtype], Array] = normal

    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.
        Args:
          inputs: The nd-array to be transformed.
        Returns:
          The transformed input.
        """
        inputs = jnp.asarray(inputs, self.dtype)
        full_dim = inputs.shape[-1] * self.features

        key = self.make_rng('kernel')
        kernel = self.variable('kernel', 'kernel',
                               self.kernel_init, key,
                               (1, full_dim))
        kernel = jnp.asarray(kernel.value, self.dtype)

        subspace_kernel = self.param('subspace_kernel', self.subspace_kernel_init, (1, self.subspace_dim))
        subspace_kernel = jnp.asarray(subspace_kernel, self.dtype)

        key = self.make_rng('projection_matrix')
        projection_matrix = self.variable('projection_matrix', 'projection_matrix',
                                          self.projection_matrix_init, key,
                                          (self.subspace_dim, full_dim))
        projection_matrix = jnp.asarray(projection_matrix.value, self.dtype)

        weight = jnp.matmul(subspace_kernel, projection_matrix) + kernel
        weight = weight.reshape((inputs.shape[-1], self.features))

        y = lax.dot_general(inputs, weight,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
        if self.use_bias:
            subspace_bias_kernel = self.param('subspace_bias', self.subspace_bias_init, (1, self.subspace_dim))
            subspace_bias_kernel = jnp.asarray(subspace_bias_kernel, self.dtype)

            key = self.make_rng('bias')
            bias = self.variable('bias', "b", self.bias_init, key, (1, self.features))
            bias = jnp.asarray(bias.value, self.dtype)

            key = self.make_rng('projection_matrix_bias')
            projection_matrix_bias = self.variable('projection_matrix_bias', 'projection_matrix_bias',
                                                   self.projection_matrix_init, key,
                                                   (self.subspace_dim, self.features))

            projection_matrix_bias = jnp.asarray(projection_matrix_bias.value, self.dtype)
            b = jnp.matmul(subspace_bias_kernel, projection_matrix_bias) + bias

            y += jnp.reshape(b, (1,) * (y.ndim - 1) + (-1,))

        return y