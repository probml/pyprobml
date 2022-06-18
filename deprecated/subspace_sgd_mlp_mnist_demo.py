# This demo replicates Figure 2 of the paper
# "Measuring the Intrinsic Dimension of Objetive Landscape"
# By Li et al. (https://arxiv.org/abs/1804.08838)
# We consider a 2-layer MLP with ReLU activations
# Code based on the following repos:
# * https://github.com/ganguli-lab/degrees-of-freedom
# * https://github.com/uber-research/intrinsic-dimension


# Author :  Gerardo Durán-Martín (@gerdm), Aleyna Kara(@karalleyna), Kevin Murphy(@murphyk)

# import superimport

import matplotlib.pyplot as plt
import pyprobml_utils as pml

import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split, permutation, normal
from jax.experimental.stax import Dense, Relu, LogSoftmax
from jax.experimental import stax

import optax
from tensorflow.keras.datasets import mnist

#from subspace_opt_lib import make_potential, make_potential_subspace, optimize_loop
import subspace_opt_lib as sub


def load_mnist(key, n_train, n_test, shuffle=True):
    (X, y), (X_test, y_test) = mnist.load_data()

    n_train = n_train if n_train < len(y) else len(y)
    n_test = n_test if n_test < len(y_test) else len(y)

    train_key, test_key = split(key)
    train_indices = jnp.arange(len(y))
    perm = permutation(train_key, train_indices)[:n_train] if shuffle else train_indices[:n_train]

    train_ds = {
        "X": jnp.float32(X[perm].reshape(n_train, -1)) / 255.,
        "y": jnp.array(y[perm])
    }

    test_indices = jnp.arange(len(y_test))
    perm = permutation(test_key, test_indices)[:n_test] if shuffle else test_indices[:n_test]

    test_ds = {
        "X": jnp.float32(X_test[perm].reshape(n_test, -1)) / 255.,
        "y": jnp.array(y_test[perm])
    }

    return train_ds, test_ds



key = PRNGKey(42)
data_key, key = split(key)

n_train, n_test = 5000, 1000
train_ds, test_ds = load_mnist(data_key, n_train, n_test)

n_features = train_ds["X"].shape[1]
n_classes = 10

init_random_params, predict = stax.serial(
    Dense(n_features), Relu,
    Dense(50), Relu,
    Dense(n_classes), LogSoftmax)

init_key, key = split(key)
_, params_tree_init = init_random_params(init_key, input_shape=(-1, n_features))

# Do one step of SGD in full parameter space to get good initial value (“anchor”)
potential_key, key = split(key)
l2_regularizer, batch_size = 1., 512
objective = sub.make_potential(potential_key, predict, train_ds, batch_size, l2_regularizer)

losses = jnp.array([])
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
n_steps = 300
anchor_params_tree, loss, _ = sub.optimize_loop(objective, params_tree_init, optimizer, n_steps=n_steps, callback=None)
print(f"Loss : {loss[-1]}")

# Do subspace optimization starting from rnd location
subspace_dim = 100
subspace_key, key = split(key)
anchor_params_full, flat_to_pytree_fn = jax.flatten_util.ravel_pytree(anchor_params_tree)
full_dim = len(anchor_params_full)
projection_matrix = sub.generate_random_basis(key, subspace_dim, full_dim)

objective_subspace, subspace_to_pytree_fn = sub.make_potential_subspace(
    subspace_key, anchor_params_tree, predict, train_ds, batch_size, l2_regularizer,
    subspace_dim, projection_matrix=projection_matrix)

losses = jnp.array([])
params_subspace = normal(key, shape=(subspace_dim,))
params_subspace, loss, _ = sub.optimize_loop(objective_subspace, params_subspace, optimizer, n_steps)
print(f"Loss : {loss[-1]}")
losses = jnp.append(losses, loss)

# Do more subspace optimization continuing from before (warm-start)
params_subspace, loss, _ = sub.optimize_loop(objective_subspace, params_subspace, optimizer, n_steps)
print(f"Loss : {loss[-1]}")
losses = jnp.append(losses, loss)

# Plot loss curve
plt.plot(losses, linewidth=3)
plt.xlabel("Iteration")
pml.savefig("subspace_sgd_mlp_mnist_demo.png")
plt.show()
