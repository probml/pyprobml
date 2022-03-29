# Backpropagation Applied to MNIST (Modern Version)
# Based on Lecun 1989: http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf
# Adapted to JAX from https://github.com/karpathy/lecun1989-repro/blob/master/prepro.py
# Author: Peter G. Chang (@peterchang0414)
import os
import argparse
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from torchvision import datasets
from jax import value_and_grad
from flax import linen as nn
from flax import serialization
from flax.training import train_state
from flax.linen.activation import tanh

def get_datasets(key, n_tr, n_te):
    train_test = {}
    for split in {'train', 'test'}:
        data = datasets.MNIST('./data', train=split=='train', download=True)
        n = n_tr if split == 'train' else n_te
        key, _ = jax.random.split(key)
        rp = jax.random.permutation(key, len(data))[:n]
        X = jnp.full((n, 16, 16, 1), 0.0, dtype=jnp.float32)
        Y = jnp.full((n, 10), -1.0, dtype=jnp.float32)
        for i, ix in enumerate(rp):
            I, yint = data[int(ix)]
            xi = jnp.array(I, dtype=jnp.float32) / 127.5 - 1.0
            xi = jax.image.resize(xi, (16, 16), 'bilinear')
            X = X.at[i].set(jnp.expand_dims(xi, axis=2))
            Y = Y.at[i, yint].set(1.0)
        train_test[split] = (X, Y)
    return train_test

class Net(nn.Module):
    training: bool
    bias_init: Callable = nn.initializers.zeros
    # sqrt(6) = 2.449... used by he_uniform() approximates Karpathy's 2.4
    kernel_init: Callable = nn.initializers.he_uniform()

    @nn.compact
    def __call__(self, x):
        if self.training:
            augment_rng = self.make_rng('aug')
            shift_x, shift_y = jax.random.randint(augment_rng, (2,), -1, 2)
            x = jnp.roll(x, (shift_x, shift_y), (1, 2))
        x = jnp.pad(x, [(0,0),(2,2),(2,2),(0,0)], constant_values=-1.0)
        x = nn.Conv(features=12, kernel_size=(5,5), strides=2, padding='VALID',
                    use_bias=False, kernel_init=self.kernel_init)(x)
        bias1 = self.param('bias1', self.bias_init, (8, 8, 12))
        x = nn.relu(x + bias1)
        x = jnp.pad(x, [(0,0),(2,2),(2,2),(0,0)], constant_values=-1.0)
        x1, x2, x3 = (x[..., 0:8], x[..., 4:12], 
                      jnp.concatenate((x[..., 0:4], x[..., 8:12]), axis=-1))
        slice1 = nn.Conv(features=4, kernel_size=(5,5), strides=2, padding='VALID', 
                         use_bias=False, kernel_init=self.kernel_init)(x1)
        slice2 = nn.Conv(features=4, kernel_size=(5,5), strides=2, padding='VALID',
                         use_bias=False, kernel_init=self.kernel_init)(x2)
        slice3 = nn.Conv(features=4, kernel_size=(5,5), strides=2, padding='VALID',
                         use_bias=False, kernel_init=self.kernel_init)(x3)
        x = jnp.concatenate((slice1, slice2, slice3), axis=-1)
        bias2 = self.param('bias2', self.bias_init, (4, 4, 12))
        x = nn.relu(x + bias2)
        x = nn.Dropout(0.25, deterministic=not self.training)(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=30, use_bias=False)(x)
        bias3 = self.param('bias3', self.bias_init, (30,))
        x = nn.relu(x + bias3)
        x = nn.Dense(features=10, use_bias=False)(x)
        bias4 = self.param('bias4', self.bias_init, (10,))
        x = x + bias4
        return x

def learning_rate_fn(initial_rate, epochs, steps_per_epoch):
    return optax.linear_schedule(init_value=initial_rate, end_value=initial_rate/3,
                                 transition_steps=epochs*steps_per_epoch)

def create_train_state(key, X, lr_fn):
    model = Net(training=True)
    key1, key2, key3 = jax.random.split(key, 3)
    params = model.init({'params': key1, 'aug': key2, 'dropout': key3}, X)['params']
    opt = optax.adamw(lr_fn)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)

@jax.jit
def train_step(state, X, Y, rng=jax.random.PRNGKey(0)):
    aug_rng, dropout_rng = jax.random.split(jax.random.fold_in(rng, state.step))
    def loss_fn(params):
        Yhat = Net(training=True).apply({'params': params}, X, 
                                        rngs={'aug': aug_rng,
                                              'dropout': dropout_rng})
        loss = jnp.mean(optax.softmax_cross_entropy(logits=Yhat, labels=Y))
        err = jnp.mean(jnp.argmax(Y, -1) != jnp.argmax(Yhat, -1)).astype(float)
        return loss, err
    (_, Yhats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

def train_one_epoch(state, X, Y):
    for step_num in range(X.shape[0]):
        x, y = jnp.expand_dims(X[step_num], 0), jnp.expand_dims(Y[step_num], 0)
        state = train_step(state, x, y)
    return state

def train(key, data, epochs, lr):
    Xtr, Ytr = data['train']
    Xte, Yte = data['test']
    lr_fn = learning_rate_fn(lr, epochs, Xtr.shape[0])
    train_state = create_train_state(key, Xtr, lr_fn)
    for epoch in range(epochs):
        print(f"epoch {epoch+1} with learning rate {lr_fn(train_state.step):.6f}")
        train_state = train_one_epoch(train_state, Xtr, Ytr)
        for split in ['train', 'test']:
            eval_split(data, split, train_state.params)
    return train_state

@jax.jit
def eval_step(params, X, Y):
    Yhat = Net(training=False).apply({'params': params}, X)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=Yhat, labels=Y))
    err = jnp.mean(jnp.argmax(Y, -1) != jnp.argmax(Yhat, -1)).astype(float)
    return loss, err

def eval_split(data, split, params):
    X, Y = data[split]
    loss, err = eval_step(params, X, Y)
    print(f"eval: split {split:5s}. loss {loss:e}. "
          f"error {err*100:.2f}%. misses: {int(err*Y.shape[0])}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits")
    parser.add_argument('--learning-rate', '-l', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--output-dir'   , '-o', type=str,   default='out/modern', help="output directory for training logs")
    args = parser.parse_args()
    print(vars(args))
    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    data = get_datasets(key1, 7291, 2007)
    state = train(key2, data, 80, args.learning_rate)
    bytes_output = serialization.to_bytes(state.params)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    with open(args.output_dir, 'wb') as f:
        f.write(bytes_output)