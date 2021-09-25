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
import optax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from jax import random
from time import time
from functools import partial


def get_datasets():
    """
    Load MNIST train and test datasets into memory
    """
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    train_ds["image"] = jnp.float32(train_ds["image"]) / 255.
    test_ds["image"] = jnp.float32(test_ds["image"]) / 255.
    return train_ds, test_ds


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(784)(x))
        x = nn.relu(nn.Dense(200)(x))
        x = nn.relu(nn.Dense(200)(x))
        x = nn.Dense(10)(x)
        return nn.log_softmax(x)


@jax.vmap
def cross_entropy_loss(logits, label):
    return -logits[label]


@jax.jit
def normal_loss(params, batch):
    logits = MLP().apply({"params": params}, batch["image"])
    logits = jax.nn.log_softmax(logits)
    loss = jnp.mean(cross_entropy_loss(logits, batch["label"]))
    return loss


@jax.jit
def normal_accuracy(params,batch):
    logits = MLP().apply({"params": params}, batch["image"])
    logits = jax.nn.log_softmax(logits)
    return jnp.mean(jnp.argmax(logits, -1) == batch["label"])


@jax.jit
def theta_to_flat_params(theta,M,flat_params0):
    return jnp.matmul(theta, M)[0] + flat_params0


def projected_loss(theta_subspace, batch, M, flat_params0, reconstruct_fn):
    """
    Project the subspace model weights onto the full model weights and
    compute the loss.
            w(theta) = w_init + A * theta
    1. Project theta_subspace ∈ R^d => theta ∈ R^D
    2. Compute loss of the model w.r.t. theta_subspace
    """
    projected_params = theta_to_flat_params(theta_subspace, M, flat_params0)
    projected_params = reconstruct_fn(projected_params)
    return normal_loss(projected_params, batch)


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


def generate_projection(key, d, D):
    M = random.normal(key, shape=(d,D))
    M = M / jnp.linalg.norm(M, axis=-1, keepdims=True)
    return M


def subspace_learning(key, model, datasets, d, hyperparams, n_epochs=300):
    key_params, key_subspace = random.split(key)
    _, num_features = train_ds["image"].shape

    x0 = jnp.zeros(num_features)
    w_init = model().init(key_params, x0)["params"]
    w_init_flat, reconstruct_fn = jax.flatten_util.ravel_pytree(w_init)

    D = len(w_init_flat)
    A = generate_projection(key_subspace, d, D)
    projected_loss_partial = partial(projected_loss, M=A,
                                     reconstruct_fn=reconstruct_fn,
                                     flat_params0=w_init_flat)
    loss_grad_wrt_theta = jax.grad(projected_loss_partial)

    theta = jnp.zeros((1, d))
    mass = jnp.zeros((1, d))
    velocity = jnp.zeros((1, d))

    for e in range(n_epochs):
        grads = loss_grad_wrt_theta(theta, datasets["train"])
        theta, mass, velocity = adam_update(grads, theta, mass, velocity, hyperparams)

        params_now = theta_to_flat_params(theta, A, w_init_flat)
        params_now = reconstruct_fn(params_now)

        epoch_loss = normal_loss(params_now, datasets["train"])
        epoch_accuracy = normal_accuracy(params_now, datasets["train"])
        if e % 100 == 0 or e == n_epochs - 1:
            end = "\n"
            epoch_val_accuracy = normal_accuracy(params_now, datasets["test"])
            val_str = f" || val acc: {epoch_val_accuracy:0.2%}"
        else:
            end = "\r"
            val_str = ""
            
        metric_str = f"epoch: {e+1:03} || acc: {epoch_accuracy:0.2%} || loss:{epoch_loss:0.2f}"
        metric_str += val_str
        print(metric_str, end=end)

    return theta, epoch_val_accuracy


if __name__ == "__main__":
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    key = random.PRNGKey(314)
    n_features = 28 ** 2
    train_ds, test_ds = get_datasets()
    train_ds["image"] = train_ds["image"].reshape(-1, n_features)
    test_ds["image"] = test_ds["image"].reshape(-1, n_features)

    datasets = {
        "train": train_ds,
        "test": test_ds,
    }

    hyperparams = {
        "lr": 1e-1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7
    }

    min_dim, max_dim = 10, 1300
    jump_size = 50
    subspace_dims = [2] + list(range(min_dim, max_dim, jump_size))

    acc_vals = []
    n_epochs = 300
    for dim in subspace_dims:
        init_time = time()
        print(f"\nTesting subpace {dim=}")
        theta, accuracy = subspace_learning(key, MLP, datasets, dim, hyperparams, n_epochs=n_epochs)
        end_time = time()
        print(f"Running time: {end_time - init_time:0.2f}s")
        acc_vals.append(accuracy)
    
    fig, ax = plt.subplots(figsize=(7, 3))
    plt.plot(subspace_dims[::2], acc_vals[::2], marker="o")
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.axhline(y=0.9, c="tab:gray", linestyle="--")
    plt.xlabel("Subspace dim $d$", fontsize=13)
    plt.ylabel("Validation accuracy", fontsize=13)
    plt.tight_layout()
    pml.savefig("subspace_learning.pdf")
