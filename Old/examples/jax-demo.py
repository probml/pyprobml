#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:18:59 2019

https://github.com/google/jax

@author: kpmurphy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import re
import sys
import time

import pandas as pd
 
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as onp # original
import scipy as oscipy # original

import jax
from jax import numpy as np
from jax import scipy, random, grad, jit



# Prevent numpy from printing too many digits
np.set_printoptions(precision=3)

#def sigmoid(x):
#    return 0.5 * (np.tanh(x / 2.) + 1)

def sigmoid(x):
    return scipy.special.expit(x)

def softmax(x, axis=None):
    return scipy.special.softmax(x, axis)

# Outputs probability of a label being true according to logistic model.
def predict(weights, inputs):
    return sigmoid(np.dot(inputs, weights))

# Training loss is the negative log-likelihood of the training labels.
def loss(weights, data):
    inputs, targets = data
    preds = predict(weights, inputs)
    label_logprobs = np.log(preds) * targets + np.log(1 - preds) * (1 - targets)
    return -np.sum(label_logprobs)

################
# Make data

D = 3
N = 10
#rng = random.PRNGKey(0)
#rng1, rng = random.split(rng)
#weights_true = random.normal(rng1, (D,))
#rng1, rng = random.split(rng)
#inputs = random.normal(rng1, (N, D))
#logits = inputs.dot(weights_true)
#rng1, rng = random.split(rng)
#targets = random.bernoulli(rng1, sigmoid(logits), (N,)) 
weights_true = onp.random.randn(D)
inputs = onp.random.randn(N, D)
logits = onp.dot(inputs, weights_true)
targets = (onp.random.rand(N) < sigmoid(logits))
weights_init = onp.random.randn(D)
train_data = (inputs, targets)

################
# Full batch SGD by hand
train_loss_fun = lambda w: loss(w, train_data)
train_gradient_fun = jit(grad(train_loss_fun))
weights = weights_init
print("Initial loss: {:0.2f}".format(loss(weights, train_data)))
lr = 0.1
max_iter = 10
print("manual SGD")
for i in range(max_iter):
    weights -= lr * train_gradient_fun(weights)
weights_final1 = weights
print("Trained loss1: {:0.2f}".format(loss(weights_final1, train_data)))

################
# Full batch SGD using optim library
#import jax.experimental.optimizers
import jax.experimental.minmax as optimizers

@jit
def step(i, opt_state):
  weights = optimizers.get_params(opt_state)
  g = train_gradient_fun(weights)
  return opt_update(i, g, opt_state)

opt_init, opt_update = optimizers.sgd(step_size=lr)
opt_state = opt_init(weights_init)
print("jax SGD")
for i in range(max_iter):
  opt_state = step(i, opt_state)
weights_final2 = optimizers.get_params(opt_state)
print("Trained loss2: {:0.2f}".format(loss(weights_final2, train_data)))
 


   
################
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
from scipy.optimize import minimize

lbfgs_memory = 10
opts = {"maxiter": max_iter, "maxcor": lbfgs_memory}
iter = itertools.count()
def callback(x):
  print("iter {}, params {}".format(next(iter), x))
  return False

print("BFGS")
result = oscipy.optimize.fmin_l_bfgs_b(train_loss_fun, weights_init, fprime = train_gradient_fun)

result = minimize(train_loss_fun, weights_init,  method='BFGS',
                  jac = train_gradient_fun, options = opts, callback = callback)
weights_final3 = result.x
print("Trained loss3: {:0.2f}".format(loss(weights_final3, train_data)))

####


#if meth == '_custom':
#        return method(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
#                      bounds=bounds, constraints=constraints,
#                      callback=callback, **options)
        
# wrap SGD into scipy.minimize format
opt_init, opt_update = optimizers.sgd(step_size=lr)
def sgd(fun, params_init, args=(), jac=None, 
        callback=None, maxiter=100, **options):
  if jac is None:
    jac = grad(fun)
  def step(i, opt_state):
    params = optimizers.get_params(opt_state)
    g = jac(params)
    return opt_update(i, g, opt_state)  #opt_update
  opt_state = opt_init(params_init)  #opt_init
  for i in range(maxiter):
    opt_state = step(i, opt_state)
    if not(callback is None):
      params = optimizers.get_params(opt_state)
      if callback(params):
        break
  result.x = optimizers.get_params(opt_state)
  return result

iter = itertools.count()
def callback(x):
  print("iter {}, params {}".format(next(iter), x))
  return False
print("wrapped SGD")
result = minimize(train_loss_fun, weights_init, args=(),  method=sgd,
                  jac = train_gradient_fun, options = opts, callback = callback)
weights_final4 = result.x
print("Trained loss4: {:0.2f}".format(loss(weights_final4, train_data)))