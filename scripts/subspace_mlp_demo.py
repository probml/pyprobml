# Based on
# * https://github.com/ganguli-lab/degrees-of-freedom
# * https://github.com/uber-research/intrinsic-dimension

import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import tensorflow_datasets as tfds
from jax import random
from flax.training import train_state


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


def create_train_state(rng, learning_rate, momentum):
    """Creates initial `TrainState`."""
    mlp = MLP()
    params = mlp.init(rng, jnp.ones((1, 28 ** 2)))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
      apply_fn=mlp.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = MLP().apply({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics


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
    return jnp.matmul(theta,M)[0] + flat_params0

# To-do: add explicit dependence on , vec0, treedef, ahd shapes_list
def projected_loss(theta_subspace, batch):
    """
    Project the subspace model weights onto the full model weights and
    compute the loss.
            w(theta) = w_init + A * theta
    1. Project theta_subspace ∈ R^d => theta ∈ R^D
    2. Compute loss of the model w.r.t. theta_subspace
    """
    projected_params = theta_to_flat_params(theta_subspace, M, flat_params0)
    projected_params = reconstruct_fn(projected_params)
    return normal_loss(projected_subspace_params, batch)


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


def generate_projection(d, D, k_nonzero=None, enforce_no_overlap_if_possible=True):
    M_random = np.random.normal(loc=0.0,scale=1.0,size=(d,D))
    if k_nonzero is None:
        M_now = M_random
    else:
        M_now = np.zeros((d,D))
        if ((k_nonzero*d <= D) and (enforce_no_overlap_if_possible == True)):
            ids_flat = np.random.choice(range(D),(k_nonzero*d),replace=False)
            ids_shaped = ids_flat.reshape([d,k_nonzero])
        elif ((k_nonzero*d <= D) and (enforce_no_overlap_if_possible == False)):
            ids_flat = np.random.choice(range(D),(k_nonzero*d),replace=True)
            ids_shaped = ids_flat.reshape([d,k_nonzero])
        else:
            ids_flat = np.random.choice(range(D),(k_nonzero*d),replace=True)
            ids_shaped = ids_flat.reshape([d,k_nonzero])
        for i in range(d):
            M_now[i,ids_shaped[i]] = M_random[i,ids_shaped[i]]
    #normalization to unit length of each basis vector
    M_now = M_now / np.linalg.norm(M_now,axis=-1,keepdims=True)  
    return M_now


n_epochs = 300
hyperparams = {
    "lr": 1e-1,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-7
}


if __name__ == "__main__":
    key = random.PRNGKey(314)

    d = 20
    theta = jnp.zeros((1, d))
    mass = jnp.zeros((1, d))
    velocity = jnp.zeros((1, d))

    print(f"Testing subpace {d=}")
    n_epochs = 0
    for e in range(n_epochs):
        grads = loss_grad_wrt_theta(theta, train_ds)
        theta, mass, velocity = adam_update(grads, theta, mass, velocity, hyperparams)
        params_now = training_utils.theta_to_paramstree(theta, M, vec0, treedef, shapes_list)
        epoch_loss = normal_loss(params_now, train_ds)
        epoch_accuracy = normal_accuracy(params_now, train_ds)
        if e % 100 == 0 or e == n_epochs - 1:
            end = "\n"
            epoch_val_accuracy = normal_accuracy(params_now, test_ds)
            val_str = f" || val acc: {epoch_val_accuracy:0.2%}"
        else:
            end = "\r"
            val_str = ""
            
        metric_str = f"epoch: {e+1:03} || acc: {epoch_accuracy:0.2%} || loss:{epoch_loss:0.2f}"
        metric_str += val_str
        print(metric_str, end=end)
