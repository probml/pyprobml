# Author : Kevin Murphy(@murphyk), Aleyna Kara(@karalleyna)
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, tree_leaves, tree_map, vmap
from jax.random import split, PRNGKey, permutation
from jax.nn import one_hot
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax

import optax

from tensorflow.keras.datasets import mnist
from sgmcmcjax.samplers import build_sgldCV_sampler

import subspace_lib as sub


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
    # logits = log_softmax(logits)
    return jnp.mean(jnp.argmax(logits, -1) == batch["y"])


@jit
def logprior(params):
    # Spherical Gaussian prior
    leaves_of_params = tree_leaves(params)
    return sum(tree_map(lambda p: jnp.sum(jax.scipy.stats.norm.logpdf(p, scale=l2_regularizer)), leaves_of_params))


key = PRNGKey(42)
data_key, init_key, opt_key, sample_key, warmstart_key = split(key, 5)

n_train, n_test = 20000, 1000
train_ds, test_ds = load_mnist(data_key, n_train, n_test)
data = (train_ds["X"], train_ds["y"])
n_features = train_ds["X"].shape[1]
n_classes = 10

# model
init_random_params, predict = stax.serial(
    Dense(200), Relu,
    Dense(50), Relu,
    Dense(n_classes), LogSoftmax)

_, params_init_tree = init_random_params(init_key, input_shape=(-1, n_features))

leaves = tree_leaves(params_init_tree)
n = 0
for i in range(len(leaves)):
    sh = leaves[i].shape
    n += np.prod(sh)
    print(f"size of parameters in leaf {i} is {sh}")
print("total nparams", n)

l2_regularizer = 0.1
batch_size = 512

subspace_dim = 200
nwarmup = 50
nsteps = 100

# optimizer
print("running optimizer in subspace")
opt = optax.adam(learning_rate=1e-1)

# initialize subspace
key, mykey = split(key)
subspace_fns, subspace_dim = sub.init_subspace_opt(
    mykey, loglikelihood, logprior, params_init_tree, subspace_dim,
    data, batch_size, nwarmup, opt, use_svd=False, pbar=False)
loglik_sub, logprior_sub, subspace_to_pytree_fn = subspace_fns
key, mykey = split(key)
params_subspace_init = jax.random.normal(mykey, (subspace_dim,))

# evaluate at rnd starting location
print("performance at start")
params_subspace = params_subspace_init
params_tree = subspace_to_pytree_fn(params_subspace)
print(params_subspace[:10])
print(f"Train accuracy : {accuracy(params_tree, train_ds)}")
print(f"Test accuracy : {accuracy(params_tree, test_ds)}")

# run optimizer from rnd starting location 
key, mykey = split(key)
params_subspace = params_subspace_init
params_tree, params_subspace, log_post_trace = sub.subspace_optimizer(
    mykey, subspace_fns, data, batch_size, nsteps, opt, params_subspace, pbar=False)

print(params_subspace[:10])
print(f"Train accuracy : {accuracy(params_tree, train_ds)}")
print(f"Test accuracy : {accuracy(params_tree, test_ds)}")

# run optimizer from previous location (on "new" data) 
print("running optimizer in subspace")
data_new = data
key, mykey = split(key)
params_tree, params_subspace, log_post_trace = sub.subspace_optimizer(
    mykey, subspace_fns, data_new, batch_size, nsteps, opt, params_subspace, pbar=False)

print(params_subspace[:10])
print(f"Train accuracy : {accuracy(params_tree, train_ds)}")
print(f"Test accuracy : {accuracy(params_tree, test_ds)}")

# run sampler from rnd starting location
print("running sampler in subspace")
params_subspace = params_subspace_init
sampler = partial(build_sgldCV_sampler, dt=1e-5)  # or any other whitejax sampler
nsamples = 200

key, mykey = split(key)
params_tree_samples, params_sub_samples = sub.subspace_sampler(
    mykey, subspace_fns, data, batch_size, nsamples, sampler, params_subspace, use_cv=True, pbar=False)

# compute posteriot predictive for a single test example using all nsamples
x = test_ds["X"][0]
params = params_tree_samples
logits = vmap(predict, in_axes=(0,None))(params, x)
print(logits.shape) # nsamples x nclasses

# vmap accuracy across the sampled parameter trees, then take average
train_accuracy = jnp.mean(vmap(accuracy, in_axes=(0, None))(params_tree_samples, train_ds))
test_accuracy = jnp.mean(vmap(accuracy, in_axes=(0, None))(params_tree_samples, test_ds))

print(f"Train accuracy : {train_accuracy}")
print(f"Test accuracy : {test_accuracy}")

# run sampler from previous location (on "new" data) 
print("running sampler in subspace")
data_new = data

key, mykey = split(key)
params_subspace = params_sub_samples[0]  # pick first sample as starting point
params_tree_samples, params_sub_samples = sub.subspace_sampler(
    mykey, subspace_fns, data_new, batch_size, nsamples, sampler, params_subspace, use_cv=True, pbar=False)

# vmap accuracy across the sampled parameter trees, then take average
train_accuracy = jnp.mean(vmap(accuracy, in_axes=(0, None))(params_tree_samples, train_ds))
test_accuracy = jnp.mean(vmap(accuracy, in_axes=(0, None))(params_tree_samples, test_ds))

print(f"Train accuracy : {train_accuracy}")
print(f"Test accuracy : {test_accuracy}")
