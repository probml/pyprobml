# -*- coding: utf-8 -*-
"""
Author: Ang Ming Liang
For further explanations refer to the following gist https://gist.github.com/Neoanarika/a339224d24affd7840a30a1064fc16ff
"""

import superimport

import jax
import jax.numpy as jnp 
from jax import lax
from jax import vmap
from jax import random
from jax import jit
import matplotlib.pyplot as plt
from tqdm import trange
import pyprobml_utils as pml

# The K number of states and size of the board
K= 10
ix = 128
iy = 128

# Initalising the key and the kernel
key = random.PRNGKey(12234)
kernel = jnp.zeros((3, 3, 1, 1), dtype=jnp.float32)
kernel += jnp.array([[0, 1, 0],
                     [1, 0,1],
                     [0,1,0]])[:, :, jnp.newaxis, jnp.newaxis]

dn = lax.conv_dimension_numbers((K, ix, iy, 1),     # only ndim matters, not shape
                                 kernel.shape,  # only ndim matters, not shape 
                                ('NHWC', 'HWIO', 'NHWC'))  # the important bit

# Creating the checkerboard
mask = jnp.indices((K, iy, ix, 1)).sum(axis=0) % 2

def checkerboard_pattern1(x):
  return mask[0, :, : , 0]

def checkerboard_pattern2(x):
  return mask[1, :, : , 0]

def make_checkerboard_pattern1():
  arr = vmap(checkerboard_pattern1, in_axes=0)(jnp.array(K*[1]))
  return jnp.expand_dims(arr, -1)

def make_checkerboard_pattern2():
  arr = vmap(checkerboard_pattern2, in_axes=0)(jnp.array(K*[1]))
  return jnp.expand_dims(arr, -1)

def sampler(K, key, logits):
  # Sample from the energy using gumbel trick
  u = random.uniform(key, shape=(K, ix, iy, 1))
  sample = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=0)
  sample = jax.nn.one_hot(sample, K, axis=0)
  return sample

def state_mat_update(mask, inverse_mask, sample, state_mat):
  # Update the state_mat using masking
  masked_sample = mask*sample 
  masked_state_mat = inverse_mask*state_mat 
  state_mat  = masked_state_mat+masked_sample
  return state_mat

def energy(state_mat, jvalue):
  # Calculate energy
  logits = lax.conv_general_dilated(state_mat, jvalue*kernel, 
                                    (1,1), 'SAME', (1,1), (1,1), dn)  
  return logits

def gibbs_sampler(key, jvalue, niter=1):
  key, key2 = random.split(key)
  
  X = random.randint(key, shape=(ix, iy), minval=0, maxval=K)
  state_mat = jax.nn.one_hot(X, K, axis=0)[:, :, :, jnp.newaxis]

  mask = make_checkerboard_pattern1()
  inverse_mask = make_checkerboard_pattern2()
  
  @jit
  def state_update(key, state_mat, mask, inverse_mask):
    logits = energy(state_mat, jvalue)  
    sample = sampler(K, key, logits)
    state_mat = state_mat_update(mask, inverse_mask, sample, state_mat)
    return state_mat

  for iter in range(niter):
    key, key2 = random.split(key2)
    state_mat = state_update(key, state_mat, mask, inverse_mask )
    mask, inverse_mask = inverse_mask, mask
      
  return jnp.squeeze(jnp.argmax(state_mat, axis=0), axis=-1)

Jvals = [1.42, 1.43, 1.44]

fig, axs = plt.subplots(1, len(Jvals), figsize=(8, 8))
for t in trange(len(Jvals)):
  arr = gibbs_sampler(key, Jvals[t], niter=8000)
  axs[t].imshow(arr, cmap='Accent', interpolation="nearest")
  axs[t].set_title(f"J = {Jvals[t]}")
pml.savefig('gibbsDemoPotts.pdf')
plt.show()
