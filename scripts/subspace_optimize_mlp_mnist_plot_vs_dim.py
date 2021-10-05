# This demo replicates Figure 2 of the paper
# "Measuring the Intrinsic Dimension of Objetive Landscape"
# By Li et al. (https://arxiv.org/abs/1804.08838)
# We consider a 2-layer MLP with ReLU activations
# Code based on the following repos:
# * https://github.com/ganguli-lab/degrees-of-freedom
# * https://github.com/uber-research/intrinsic-dimension

# Author: Gerardo Durán-Martín (@gerdm), Kevin Murphy(@murphyk), Aleyna Kara(@karalleyna)

import jax
import jax.numpy as jnp
from jax import jit, tree_leaves, tree_map
from jax.random import split
from jax.nn import log_softmax, one_hot
from jax import random

import flax.linen as nn
import optax

import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt

from time import time
from functools import partial

import subspace_lib as sub


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


@jit
def loglikelihood(params, x, y):
    logits = predict(params, x)
    num_classes = logits.shape[-1]
    labels = one_hot(y, num_classes)
    ll = jnp.sum(labels * logits, axis=-1)
    return ll


@jit
def logprior(params):
    # Spherical Gaussian prior
    leaves_of_params = tree_leaves(params)
    return sum(tree_map(lambda p: jnp.sum(jax.scipy.stats.norm.logpdf(p, scale=l2_regularizer)), leaves_of_params))


@partial(jit, static_argnames=("predict_fn"))
def accuracy(params, batch, predict_fn):
    logits = predict_fn(params, batch["X"])
    logits = log_softmax(logits)
    return jnp.mean(jnp.argmax(logits, -1) == batch["y"])


plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

key = random.PRNGKey(314)

train_ds, test_ds = get_datasets()
n_data, *n_features = train_ds["X"].shape
n_features = np.prod(n_features)
train_ds["X"] = train_ds["X"].reshape(-1, n_features)
test_ds["X"] = test_ds["X"].reshape(-1, n_features)
train_data = (train_ds["X"], train_ds["y"])

hyperparams = {
    "learning_rate": 1e-1,
    "b1": 0.9,
    "b2": 0.999,
    "eps": 1e-7
}
optimizer = optax.adam(**hyperparams)

params_key, opt_key, subspace_key = split(key, 3)
params_init_tree = MLP().init(key, train_ds["X"][[0], :])["params"]

predict = lambda params, x: MLP().apply({"params": params}, x)

l2_regularizer = 0.011
#batch_size = n_data
batch_size = 512

min_dim, max_dim = 10, 800
jump_size = 200  
subspace_dims = range(min_dim, max_dim+jump_size, jump_size)

accuracy_trace = []
nwarmup = 0 # use random subsapce
nsteps = 300

for subspace_dim in subspace_dims:
    opt = optax.adam(**hyperparams)
    init_time = time()
    print(f"\nTesting subpace {subspace_dim}")
    params_tree, params_subspace, log_post_trace, subspace_fns = sub.subspace_optimizer(
        opt_key, loglikelihood, logprior, params_init_tree,
        train_data, batch_size, subspace_dim, nwarmup, nsteps, opt)
    end_time = time()
    print(f"Running time: {end_time - init_time:0.2f}s")
    test_accuracy = accuracy(params_tree, test_ds, predict)
    print(f"Test Accuracy : {test_accuracy}")
    accuracy_trace.append(test_accuracy)

fig, ax = plt.subplots(figsize=(6, 3))
plt.plot(subspace_dims[::2], accuracy_trace[::2], marker="o")
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
plt.axhline(y=0.9, c="tab:gray", linestyle="--")
plt.xlabel("Subspace dim $d$", fontsize=13)
plt.ylabel("Validation accuracy", fontsize=13)
plt.tight_layout()
plt.savefig("subspace_optimize_mlp_mnist.png")
plt.show()
