# -*- coding: utf-8 -*-

import time 
import itertools
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
from numpy.random import multivariate_normal as mvn

import jax.numpy as jnp
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental.optimizers import optimizer
from jax.experimental import stax
from jax.experimental.stax import Dense, Softplus
from jax import tree_util
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)
from jax.flatten_util import ravel_pytree

def mean(x):
    return np.zeros_like(x)

def kernel(x1,x2=None):
    x2 = x1 if x2 is None else x2
    sigma, theta = 5, 0.25
    
    dist = x1[:,None]-x2[None,:]
    return sigma**2*np.exp(-(dist)**2/(2*theta))

def sample_gp(x, epsilon=0.01):
    mu, Sigma = mean(x), kernel(x)
    return mvn(mu, Sigma + epsilon*np.eye(len(x)))

@jit
def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2) / 2.0

def data_stream():
  rng = npr.RandomState(0)
  while True:
    perm = rng.permutation(num_instances)
    for i in range(num_batches):
      batch_idx = perm[i * batch_size:(i + 1) * batch_size]
      yield X[batch_idx], y[batch_idx]
batches = data_stream()

def pgd(alpha, lambd):
  step_size = alpha
  
  def init(w0):
    return w0

  def soft_thresholding(z, threshold):
    return jnp.sign(z) * jnp.maximum(jnp.absolute(z) -  threshold , 0.0)
 
  def update(i, g, w):
    g_flat, unflatten = ravel_pytree(g)
    w_flat = ravel_pytree_jit(w)
    updated_params = soft_thresholding(w_flat - step_size * g_flat, step_size * lambd)
    return unflatten(updated_params)
  
  def get_params(w):
    return w

  def  set_step_size(lr):
    step_size = lr

  return init, update, get_params, soft_thresholding, set_step_size

ravel_pytree_jit = jit(lambda tree: ravel_pytree(tree)[0])

@jit
def line_search(w, g, batch, beta):
  lr_i = 1
  g_flat, unflatten_g = ravel_pytree(g)
  w_flat = ravel_pytree_jit(w)
  z_flat = soft_thresholding(w_flat - lr_i*g_flat, lr_i* lambd)
  z = unflatten_g(z_flat)
  for i in range(10):
    is_converged = loss(z, batch) > loss(w, batch) + g_flat@(z_flat - w_flat) + np.sum((z_flat - w_flat)**2)/(2*lr_i)
    lr_i = jnp.where(is_converged,lr_i, beta*lr_i)
  return lr_i

@jit
def update(i, opt_state, batch):
  params = get_params(opt_state)
  g = grad(loss)(params, batch)
  lr_i = line_search(params, g, batch, 0.5)
  set_step_size(lr_i)
  return opt_update(i, grad(loss)(params, batch), opt_state)

num_epochs = 60000
num_instances, num_vars  = 200, 2
batch_size, num_batches = num_instances, 1
minim, maxim = -5, 5

x = np.sort(np.random.uniform(minim, maxim, num_instances))
y  = sample_gp(x)[:,None]
x = x.reshape((-1,1))
X = np.c_[np.ones_like(x), x]

init_random_params, predict = stax.serial(
    Dense(5), Softplus,
    Dense(5), Softplus,
    Dense(5), Softplus,
    Dense(5), Softplus,
    Dense(1))

rng = random.PRNGKey(0)
lambd, step_size = 1e-1, 1e-4
opt_init, opt_update, get_params, soft_thresholding, set_step_size =  pgd(step_size,lambd)
_, init_params = init_random_params(rng, (-1, num_vars))
opt_state = opt_init(init_params)
itercount = itertools.count()

for epoch in range(num_epochs):
  start_time = time.time()
  opt_state = update(next(itercount), opt_state, next(batches))
  epoch_time = time.time() - start_time

labels = {"training" : "Data", "test" : "Deep Neural Net" }
x_test = np.arange(minim, maxim, 1e-4)
x_test = np.c_[np.ones((x_test.shape[0],1)), x_test]

params = get_params(opt_state)
plt.scatter(X[:,1], y, c='k', s=13, label=labels["training"])
plt.plot(x_test[:,1], predict(params, x_test), 'g-',linewidth=3, label=labels["test"])
plt.legend()
plt.show()
