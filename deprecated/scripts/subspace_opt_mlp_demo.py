# This demo replicates Figure 2 of the paper
# "Measuring the Intrinsic Dimension of Objetive Landscape"
# By Li et al. (https://arxiv.org/abs/1804.08838)
# We consider a 2-layer MLP with ReLU activations
# Code based on the following repos:
# * https://github.com/ganguli-lab/degrees-of-freedom
# * https://github.com/uber-research/intrinsic-dimension

# Author: Gerardo Durán-Martín (@gerdm), Kevin Murphy(@murphyk), Aleyna Kara(@karalleyna)


from jax import jit
from jax.random import split, normal
from jax.nn import log_softmax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from jax import random

from time import time
from functools import partial

from subspace_opt_lib import make_potential_subspace, optimize_loop
from subspace_mlp_demo import get_datasets


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(784)(x))
        x = nn.relu(nn.Dense(200)(x))
        x = nn.relu(nn.Dense(200)(x))
        x = nn.Dense(10)(x)
        return nn.log_softmax(x)


@partial(jit, static_argnames=("predict_fn"))
def accuracy(params, batch, predict_fn):
    logits = predict_fn(params, batch["X"])
    logits = log_softmax(logits)
    return jnp.mean(jnp.argmax(logits, -1) == batch["y"])


def callback(params_subspace, step, train_ds, test_ds, predict_fn, subspace_to_pytree_fn):
    params_pytree = subspace_to_pytree_fn(params_subspace)
    train_accuracy = accuracy(params_pytree, train_ds, predict_fn)
    test_accuracy = accuracy(params_pytree, test_ds, predict_fn)
    return train_accuracy, test_accuracy


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
n_data, *n_features = train_ds["X"].shape
n_features = np.prod(n_features)
train_ds["X"] = train_ds["X"].reshape(-1, n_features)
test_ds["X"] = test_ds["X"].reshape(-1, n_features)

hyperparams = {
    "learning_rate": 1e-2,
    "b1": 0.9,
    "b2": 0.999,
    "eps": 1e-7
}
optimizer = optax.adam(**hyperparams)

params_key, subspace_key, init_key = split(key, 3)
anchor_params_tree = MLP().init(params_key, jnp.zeros((n_features)))["params"]

predict_fn = lambda params, x: MLP().apply({"params": params}, x)

l2_regularizer = 1.0

min_dim, max_dim = 10, 1000
jump_size = 200  # 100
subspace_dims = [2] + list(range(min_dim, max_dim, jump_size))

acc_vals = []
n_steps = 100
print_every = 100

for subspace_dim in subspace_dims:
    objective, subspace_to_pytree_fn = make_potential_subspace(subspace_key, anchor_params_tree, predict_fn,
                                                               train_ds, n_data, l2_regularizer,
                                                               subspace_dim, projection_matrix=None)
    params_subspace = normal(key, shape=(subspace_dim,))
    callback_partial = partial(callback, train_ds=train_ds, test_ds=test_ds, predict_fn=predict_fn,
                               subspace_to_pytree_fn=subspace_to_pytree_fn)
    init_time = time()
    print(f"\nTesting subpace {subspace_dim}")
    params, loss_values, callback_hist = optimize_loop(objective, params_subspace, optimizer, n_steps,
                                                       callback=callback_partial)
    end_time = time()
    print(f"Running time: {end_time - init_time:0.2f}s")

    train_accuracies, test_acccuracies = callback_hist
    acc_vals.append(test_acccuracies[-1])
    print_metrics(loss_values, train_accuracies, test_acccuracies, print_every=100)

fig, ax = plt.subplots(figsize=(6, 3))
plt.plot(subspace_dims[::2], acc_vals[::2], marker="o")
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
plt.axhline(y=0.9, c="tab:gray", linestyle="--")
plt.xlabel("Subspace dim $d$", fontsize=13)
plt.ylabel("Validation accuracy", fontsize=13)
plt.tight_layout()
pml.savefig("subspace_mlp_demo.png")
plt.show()
