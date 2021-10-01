# Author : Kevin Murphy(@murphyk), Aleyna Kara(@karalleyna)

import matplotlib.pyplot as plt
import pyprobml_utils as pml
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, tree_leaves, tree_map
from jax.random import split, PRNGKey, permutation
from jax.nn import one_hot, log_softmax
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax

import optax

from tensorflow.keras.datasets import mnist
from sgmcmcjax.samplers import build_sgldCV_sampler

import sgmcmc_subspace_lib as sub


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


# objective
@jit
def loglikelihood(params, x, y):
    logits = predict(params, x)
    num_classes = logits.shape[-1]
    labels = one_hot(y, num_classes)
    ll = jnp.sum(labels * logits, axis=-1)
    return ll


@jit
def accuracy(params, batch):
    logits = predict(params, batch["X"])
    logits = log_softmax(logits)
    return jnp.mean(jnp.argmax(logits, -1) == batch["y"])


@jit
def logprior(params):
    # Spherical Gaussian prior
    leaves_of_params = tree_leaves(params)
    return sum(tree_map(lambda p: jnp.sum(jax.scipy.stats.norm.logpdf(p, scale=l2_regularizer)), leaves_of_params))


key = PRNGKey(42)
data_key, init_key, opt_key, sample_key, warmstart_key = split(key, 5)

n_train, n_test = 5000, 1000
train_ds, test_ds = load_mnist(data_key, n_train, n_test)
data = (train_ds["X"], train_ds["y"])
n_features = train_ds["X"].shape[1]
n_classes = 10

subspace_dim = 100
nwarmup = 100
nsteps = 300
nsamples = 300

# model
init_random_params, predict = stax.serial(
    Dense(n_features), Relu,
    Dense(50), Relu,
    Dense(n_classes), LogSoftmax)

_, params_init_tree = init_random_params(init_key, input_shape=(-1, n_features))

l2_regularizer = 0.01
batch_size = 512

# optimize
opt = optax.adam(learning_rate=1e-3)
params_tree, params_subspace, prev_log_post_trace, loglik_sub, logprior_sub, subspace_to_pytree_fn = sub.subspace_optimizer(
    opt_key, loglikelihood, logprior, params_init_tree, data, batch_size, subspace_dim, nwarmup, nsteps, opt)

print(f"Train accuracy : {accuracy(params_tree, train_ds)}")
print(f"Test accuracy : {accuracy(params_tree, test_ds)}")

# optimize then sample
sampler = partial(build_sgldCV_sampler, dt=1e-12)  # or any other whitejax sampler
params_tree_samples = sub.subspace_sampler(sample_key, loglikelihood, logprior, params_init_tree, sampler, data,
                                           batch_size, subspace_dim, nwarmup, nsteps, nsamples, use_cv=True, opt=opt)
params_tree_samples_mean = tree_map(lambda x: jnp.mean(x, axis=0), params_tree_samples)

print(f"Train accuracy : {accuracy(params_tree_samples_mean, train_ds)}")
print(f"Test accuracy : {accuracy(params_tree_samples_mean, test_ds)}")

# Do more subspace optimization continuing from before (warm-start)
optimizer_sub = sub.build_optax_optimizer(opt, loglik_sub, logprior_sub, data, batch_size)
_, log_post_trace = optimizer_sub(warmstart_key, nsteps, params_subspace)

# Loss Curve
plt.plot(-jnp.append(prev_log_post_trace, log_post_trace), linewidth=2)
pml.savefig("subspace_sgd_mlp_mnist_demo.png")
plt.show()
