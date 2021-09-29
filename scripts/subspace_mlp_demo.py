'''
This demo replicates Figure 2 of the paper
    "Measuring the Intrinsic Dimension of Objetive Landscape"
    By Li et al. (https://arxiv.org/abs/1804.08838)
We consider a 2-layer MLP with ReLU activations
Author: Gerardo Durán-Martín (@gerdm), Aleyna Kara(@karalleyna)
'''

import superimport

import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from time import time

import tensorflow_datasets as tfds

import jax.numpy as jnp
from jax.random import PRNGKey, split
import flax.linen as nn

from subspace_learning_lib import subspace_learning, loglikelihood, logprior, accuracy
from subspace_learning_lib import build_sgd, build_nuts_sampler

import pyprobml_utils as pml

def get_datasets():
    """
    Load MNIST train and test datasets into memory
    """
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))

    train_ds["X"] = train_ds.pop("image")
    train_ds["y"] = train_ds.pop("label")

    test_ds["X"] = test_ds.pop("image")
    test_ds["y"] = test_ds.pop("label")

    train_ds["X"] = jnp.float32(train_ds["X"]) / 255.
    test_ds["X"] = jnp.float32(test_ds["X"]) / 255.
    return train_ds, test_ds


plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

train_ds, test_ds = get_datasets()

num_train, *num_features = train_ds["X"].shape
num_features = np.product(num_features)
num_classes = 10

train_ds["X"] = train_ds["X"].reshape(-1, num_features)
test_ds["X"] = test_ds["X"].reshape(-1, num_features)

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(num_features)(x))
        x = nn.relu(nn.Dense(200)(x))
        x = nn.relu(nn.Dense(200)(x))
        x = nn.Dense(num_classes)(x)
        return nn.log_softmax(x)

model = MLP()

predict_fn = lambda params, x, model : model.apply({"params": params}, x)
init_random_params_fn = lambda key, input_shape, model: (None, model.init(key, jnp.ones((input_shape[1])))["params"])

predict = partial(predict_fn, model=model)
init_random_params = partial(init_random_params_fn, model=model)

key = PRNGKey(42)
init_key, key = split(key)
_, params_full_init = init_random_params(init_key, input_shape=(-1, num_features))

hyperparams = {"batch_size": num_train,"optimizer_params" : { "lr": 1e-2, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7 } }

min_dim, max_dim = 10, 1300
jump_size = 50
subspace_dims = [2] + list(range(min_dim, max_dim, jump_size))

acc_vals = []
n_epochs = 300
num_samples = 500
num_warmup = 500

run_key, key = split(key)
for dim in subspace_dims:
    init_time = time()
    print(f"\nTesting subpace {dim}")
    params_subspace = subspace_learning(run_key, num_samples, num_warmup, train_ds, dim, \
                                  loglikelihood, logprior, params_full_init, \
                                  build_sgd, predict, hyperparams.copy())
    end_time = time()
    print(f"Running time: {end_time - init_time:0.2f}s")
    train_acc, _ = accuracy(params_subspace, (train_ds["X"], train_ds["y"]), predict)
    print(f"Training Accuracy : {train_acc}")
    acc_vals.append(train_acc)

fig, ax = plt.subplots(figsize=(6, 3))
plt.plot(subspace_dims[::2], acc_vals[::2], marker="o")
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
plt.axhline(y=0.9, c="tab:gray", linestyle="--")
plt.xlabel("Subspace dim $d$", fontsize=13)
plt.ylabel("Validation accuracy", fontsize=13)
plt.tight_layout()
pml.savefig("subspace_learning.pdf")