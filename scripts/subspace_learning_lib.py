'''
This library implements the necessary functions for subspace learning.

Code based on the following repos:
    * https://github.com/ganguli-lab/degrees-of-freedom
    * https://github.com/uber-research/intrinsic-dimension

Note that we project the subspace model weights onto the full model weights in order to
compute the loss. The steps can be summarized as follows
        w(theta) = w_init + A * theta
1. Project theta_subspace ∈ R^d => theta ∈ R^D
2. Reconstruct the pytree of the reconstructed weights
3. Compute loss of the model w.r.t. theta_subspace

Author: Gerardo Durán-Martín (@gerdm), Aleyna Kara(@karalleyna)
'''

import superimport

import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split, normal
from jax import vmap, jit, value_and_grad, tree_map
from jax.lax import scan

from blackjax import nuts, stan_warmup
from sgmcmcjax.util import progress_bar_scan



@jit
def convert_params_from_subspace_to_full(params_subspace, projection_matrix, params_full_init):
    """
    Project the subspace model weights onto the full model weights
    Parameters
    ----------
    params_subspace: jnp.ndarray(d)
        The subspace model weights
    projection_matrix: jnp.ndarray(D,d)
        The projection matrix
    params_full_init: jnp.ndarray(D)
        The initial full model weights
    Returns
    -------
    params_full: jnp.ndarray(D)
    """
    params_full = jnp.matmul(params_subspace, projection_matrix)[0] + params_full_init
    return params_full


def generate_random_basis(key, d, D):
    projection_matrix = normal(key, shape=(d, D))
    projection_matrix = projection_matrix / jnp.linalg.norm(projection_matrix, axis=-1, keepdims=True)
    return projection_matrix


@jit
def adam_update(grads, params, mass, velocity, hyperparams):
    mass = hyperparams["beta_1"] * mass + (1.0 - hyperparams["beta_1"]) * grads
    velocity = hyperparams["beta_2"] * velocity + (1.0 - hyperparams["beta_2"]) * (grads ** 2.0)
    # Bias correction
    hat_mass = mass / (1 - hyperparams["beta_1"])
    hat_velocity = velocity / (1 - hyperparams["beta_2"])
    # Update
    params = params - hyperparams["lr"] / (jnp.sqrt(hat_velocity) + hyperparams["epsilon"]) * hat_mass
    return params, mass, velocity


def loglikelihood(params, x, y, predict_fn):
    """Computes the log-likelihood."""
    logits = predict_fn(params, x)
    ll = jnp.sum(logits[y])
    return ll


@jit
def logprior(params_full, prior_variance=1.):
    """Computes the Gaussian prior log-density."""
    exp_term = -params_full ** 2 / (2 * prior_variance)
    norm_constant = 0.5 * jnp.log((2 * jnp.pi * prior_variance))
    return (exp_term - norm_constant).mean()


def accuracy(params, batch, predict_fn):
    inputs, targets = batch
    logits = predict_fn(params, inputs)
    predicted_class = jnp.argmax(logits, axis=-1)
    return jnp.mean(predicted_class == targets), jnp.exp(logits)


def data_stream(dataset, batch_size):
    X, y = dataset
    num_train = len(X)
    num_batches, rem = divmod(num_train, batch_size)
    num_batches += 1 if rem else 0
    rng = np.random.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:min((i + 1) * batch_size, num_train)]
            yield X[batch_idx], y[batch_idx]


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @progress_bar_scan(num_samples)
    def one_step(carry, i):
        state, key = carry
        kernel_key, key = split(key)
        state, _ = kernel(kernel_key, state)
        return (state, key), state

    _, states = scan(one_step, (initial_state, rng_key), jnp.arange(num_samples))
    return states


def train_step(carry, i, dataloader, loss, hyperparams, callback):
    batch = next(dataloader)
    params, mass, velocity = carry
    epoch_loss, grads = value_and_grad(loss)(params, batch)
    params, mass, velocity = adam_update(grads, params, mass, velocity, hyperparams)
    epoch_accuracy = callback(params, batch)
    return (params, mass, velocity), (epoch_loss, epoch_accuracy)


def build_nuts_sampler(num_warmup, potential):
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
        states = inference_loop(inference_key, nuts_kernel, final_state, num_samples)
        return states.position

    return nuts_sampler


def build_sgd(train_step, dataset, batch_size):
    dataloader = data_stream(dataset, batch_size)
    train_step_partial = partial(train_step, dataloader=dataloader)

    def sgd(_, num_samples, params):
        iterations = jnp.arange(num_samples)
        mass = jnp.zeros(params.shape)
        velocity = jnp.zeros(params.shape)

        @progress_bar_scan(num_samples)
        def train_step_(*args, **kwargs):
            return train_step_partial(*args, **kwargs)

        (final_params, _, _), _ = scan(train_step_, (params, mass, velocity), iterations)
        return final_params[None, ...]

    return sgd


def subspace_learning(key, num_samples, num_warmup, dataset, d, loglikelihood_fn, logprior_fn, params, sampler_init_fn,
                      predict_fn, hyperparams):
    initial_full_params, flat_to_pytree_fn = jax.flatten_util.ravel_pytree(params)
    D = len(initial_full_params)
    subspace_key, key = split(key)
    projection_matrix = generate_random_basis(subspace_key, d, D)

    def subspace_train_callback(params_subspace, dataset):
        params_full = convert_params_from_subspace_to_full(params_subspace, projection_matrix, initial_full_params)
        params_pytree = flat_to_pytree_fn(params_full)
        epoch_accuracy, _ = partial(accuracy, predict_fn=predict_fn)(params_pytree, dataset)
        return epoch_accuracy

    def projected_loglikelihood(params_subspace, *data):
        params_full = convert_params_from_subspace_to_full(params_subspace, projection_matrix, initial_full_params)
        params_pytree = flat_to_pytree_fn(params_full)
        return partial(loglikelihood_fn, predict_fn=predict_fn)(params_pytree, *data)

    def projected_logprior(params_subspace):
        params_full = convert_params_from_subspace_to_full(params_subspace, projection_matrix, initial_full_params)
        return logprior_fn(params_full)

    def projected_potential(params_subspace, batch):
        X, y = batch
        params_full = convert_params_from_subspace_to_full(params_subspace, projection_matrix, initial_full_params)
        params_pytree = flat_to_pytree_fn(params_full)
        batch_loglikelihood = vmap(loglikelihood_fn, in_axes=(None, 0, 0, None))(params_pytree, X, y, predict_fn)
        return -jnp.sum(batch_loglikelihood) - projected_logprior(params_subspace)

    if "centering_value" in hyperparams and hyperparams["centering_value"] is None:
        optimizer_params = hyperparams.pop("optimizer_params") if "optimizer_params" in hyperparams \
            else {"lr": 1e-1, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7}

        train_step_partial = partial(train_step,
                                     loss=projected_potential,
                                     hyperparams=optimizer_params,
                                     callback=subspace_train_callback)

        sampler = build_sgd(batch_size=len(dataset["X"]), train_step=train_step_partial,
                            dataset=(dataset["X"], dataset["y"]))

        initial_subspace_params = jnp.zeros((1, d))
        hyperparams["centering_value"] = sampler(None, num_warmup, initial_subspace_params)[0]
        num_warmup = 0

    if not hyperparams:
        sampler = sampler_init_fn(num_warmup=num_warmup,
                                  potential=partial(projected_potential, batch=(dataset["X"], dataset["y"])))
        num_warmup = 0

    elif "optimizer_params" in hyperparams:
        optimizer_params = hyperparams.pop("optimizer_params") if "optimizer_params" in hyperparams \
            else {"lr": 1e-1, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7}
        train_step_partial = partial(train_step, loss=projected_potential, hyperparams=optimizer_params,
                                     callback=subspace_train_callback)

        sampler = sampler_init_fn(**hyperparams, train_step=train_step_partial, dataset=(dataset["X"], dataset["y"]))
        num_samples += num_warmup
        num_warmup = 0
    else:
        sampler = sampler_init_fn(**hyperparams, loglikelihood=projected_loglikelihood, \
                                  logprior=projected_logprior, data=(dataset["X"][:, None, ...], dataset["y"]))
        num_samples += num_warmup

    initial_subspace_params = hyperparams["centering_value"] if "centering_value" in hyperparams else jnp.zeros((1, d))
    sample_key, key = split(key)
    sampler_output = sampler(sample_key, num_samples, initial_subspace_params)

    final_subspace_params = tree_map(lambda x: x[num_warmup:].mean(axis=0), sampler_output)
    final_full_params = convert_params_from_subspace_to_full(final_subspace_params, projection_matrix,
                                                             initial_full_params)

    return flat_to_pytree_fn(final_full_params)