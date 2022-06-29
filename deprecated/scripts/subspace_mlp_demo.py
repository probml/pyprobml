# This demo replicates Figure 2 of the paper
# "Measuring the Intrinsic Dimension of Objetive Landscape"
# By Li et al. (https://arxiv.org/abs/1804.08838)
# We consider a 2-layer MLP with ReLU activations
# Code based on the following repos:
# * https://github.com/ganguli-lab/degrees-of-freedom
# * https://github.com/uber-research/intrinsic-dimension

# Author: Gerardo Durán-Martín (@gerdm)

import superimport

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from jax import random
from time import time
from jax.flatten_util import ravel_pytree
from functools import partial


def get_datasets():
    """
    Load MNIST train and test datasets into memory
    """
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))

    train_ds["X"] = train_ds.pop("image")
    train_ds["y"] = jnp.array(train_ds.pop("label"))

    test_ds["X"] = test_ds.pop("image")
    test_ds["y"] = jnp.array(test_ds.pop("label"))

    train_ds["X"] = jnp.float32(train_ds["X"]) / 255.
    test_ds["X"] = jnp.float32(test_ds["X"]) / 255.

    return train_ds, test_ds


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(784)(x))
        x = nn.relu(nn.Dense(200)(x))
        x = nn.relu(nn.Dense(200)(x))
        x = nn.Dense(10)(x)
        return nn.log_softmax(x)


@jax.jit
def cross_entropy_loss(params, batch):
    logits = MLP().apply({"params": params}, batch["X"])
    logits = jax.nn.log_softmax(logits)
    loss = jnp.mean(-logits[batch["y"]])
    return loss


@jax.jit
def normal_accuracy(params, batch):
    logits = MLP().apply({"params": params}, batch["X"])
    logits = jax.nn.log_softmax(logits)
    return jnp.mean(jnp.argmax(logits, -1) == batch["y"])


@jax.jit
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


def projected_loss(params_subspace, batch, projection_matrix, params_full_init, flat_to_pytree_fn):
    """
    Project the subspace model weights onto the full model weights and
    compute the loss.
            w(theta) = w_init + A * theta
    1. Project theta_subspace ∈ R^d => theta ∈ R^D
    2. Reconstruct the pytree of the reconstructed weights
    3. Compute loss of the model w.r.t. theta_subspace
    Parameters
    ----------
    theta_subspace: jnp.ndarray(d)
        The subspace model weights
    bath: dict
        The batch of data to train
    projection_matrix: jnp.ndarray(D,d)
        The projection matrix
    params_full_init: jnp.ndarray(D)
        The initial full model weights
    flat_to_pytree_fn: function
        The reconstruction function from array(D) to pytree

    Returns
    -------
    loss: float
    """
    params_full = convert_params_from_subspace_to_full(params_subspace, projection_matrix, params_full_init)
    params_pytree = flat_to_pytree_fn(params_full)
    return cross_entropy_loss(params_pytree, batch)


@jax.jit
def adam_update(grads, params, mass, velocity, hyperparams):
    mass = hyperparams["beta_1"] * mass + (1.0 - hyperparams["beta_1"]) * grads
    velocity = hyperparams["beta_2"] * velocity + (1.0 - hyperparams["beta_2"]) * (grads ** 2.0)
    # Bias correction
    hat_mass = mass / (1 - hyperparams["beta_1"])
    hat_velocity = velocity / (1 - hyperparams["beta_2"])
    # Update
    params = params - hyperparams["lr"] / (jnp.sqrt(hat_velocity) + hyperparams["epsilon"]) * hat_mass
    return params, mass, velocity


def generate_random_basis(key, d, D):
    projection_matrix = random.normal(key, shape=(d, D))
    projection_matrix = projection_matrix / jnp.linalg.norm(projection_matrix, axis=-1, keepdims=True)
    return projection_matrix


def subspace_learning(key, model, datasets, d, hyperparams, n_epochs=300):
    key_params, key_subspace = random.split(key)
    _, num_features = train_ds["X"].shape

    x0 = jnp.zeros(num_features)
    params_full_init = model().init(key_params, x0)["params"]
    params_full_init, flat_to_pytree_fn = ravel_pytree(params_full_init)

    D = len(params_full_init)
    projection_matrix = generate_random_basis(key_subspace, d, D)
    projected_loss_partial = partial(projected_loss, projection_matrix=projection_matrix,
                                     flat_to_pytree_fn=flat_to_pytree_fn,
                                     params_full_init=params_full_init)
    loss_grad_wrt_params_subspace = jax.grad(projected_loss_partial)

    def train_step(params, i):
        params_subspace, mass, velocity = params
        grads = loss_grad_wrt_params_subspace(params_subspace, datasets["train"])
        params_subspace, mass, velocity = adam_update(grads, params_subspace, mass, velocity, hyperparams)

        params_full = convert_params_from_subspace_to_full(params_subspace, projection_matrix, params_full_init)
        params_pytree = flat_to_pytree_fn(params_full)

        epoch_loss = cross_entropy_loss(params_pytree, datasets["train"])
        epoch_accuracy = normal_accuracy(params_pytree, datasets["train"])
        epoch_val_accuracy = normal_accuracy(params_pytree, datasets["test"])
        return (params_subspace, mass, velocity), (epoch_loss, epoch_accuracy, epoch_val_accuracy)

    initial_params_subspace = jnp.zeros((1, d))
    mass = jnp.zeros((1, d))
    velocity = jnp.zeros((1, d))
    epochs = jnp.arange(n_epochs)

    (params_subspace, _, _), (loss_values, train_accuracies, val_accuracies) = jax.lax.scan(train_step, (
        initial_params_subspace, mass, velocity), epochs)

    return params_subspace, loss_values, train_accuracies, val_accuracies


def print_metrics(loss_values, train_accuracies, val_accuracies, print_every=100):
    n_epochs = loss_values.size

    metric_str = lambda e, epoch_loss, epoch_train_accuracy, epoch_val_accuracy: \
        "epoch: {:03} || loss:{:.2f} || acc: {:.2%} || val acc: {:.2%}".format(e, epoch_loss, epoch_train_accuracy,
                                                                               epoch_val_accuracy)

    for e in range(0, n_epochs, print_every):
        epoch_loss, epoch_train_accuracy, epoch_val_accuracy = loss_values[e], train_accuracies[e], val_accuracies[e]
        print(metric_str(e + 1, epoch_loss, epoch_train_accuracy, epoch_val_accuracy))

    if e != n_epochs - 1:
        epoch_loss, epoch_train_accuracy, epoch_val_accuracy = loss_values[n_epochs - 1], train_accuracies[
            n_epochs - 1], val_accuracies[n_epochs - 1]
        print(metric_str(n_epochs, epoch_loss, epoch_train_accuracy, epoch_val_accuracy))


plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

key = random.PRNGKey(314)
train_ds, test_ds = get_datasets()
_, *n_features = train_ds["X"].shape
n_features = np.prod(n_features)
train_ds["X"] = train_ds["X"].reshape(-1, n_features)
test_ds["X"] = test_ds["X"].reshape(-1, n_features)

datasets = {
    "train": train_ds,
    "test": test_ds,
}

hyperparams = {
    "lr": 1e-2,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-7
}

min_dim, max_dim = 10, 1000
jump_size = 200  # 100
subspace_dims = [2] + list(range(min_dim, max_dim, jump_size))

acc_vals = []
n_epochs = 100

for dim in subspace_dims:
    init_time = time()
    print(f"\nTesting subpace {dim}")
    params_subspace, loss_values, train_accuracies, val_accuracies = subspace_learning(key, MLP, datasets, dim,
                                                                                       hyperparams, n_epochs=n_epochs)
    end_time = time()
    print(f"Running time: {end_time - init_time:0.2f}s")
    acc_vals.append(val_accuracies[-1])
    print_metrics(loss_values, train_accuracies, val_accuracies, print_every=100)

fig, ax = plt.subplots(figsize=(6, 3))
plt.plot(subspace_dims[::2], acc_vals[::2], marker="o")
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
plt.axhline(y=0.9, c="tab:gray", linestyle="--")
plt.xlabel("Subspace dim $d$", fontsize=13)
plt.ylabel("Validation accuracy", fontsize=13)
plt.tight_layout()
plt.savefig("subspace_mlp_demo.png")
plt.show()
