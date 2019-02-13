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

import jax
from jax import lax
from jax import numpy as np
from jax import scipy
from jax import random
from jax import grad, jit

import numpy as onp # original
import scipy as oscipy # original


key = random.PRNGKey(0)
N = 500
x = random.normal(key, (N, N))
print(np.dot(x, x.T) / 2)  # fast!
print(np.dot(x, x.T) / 2)  # even faster!



#def sigmoid(x):
#    return 0.5 * (np.tanh(x / 2.) + 1)

def sigmoid(x):
    return scipy.special.expit(x)

# Outputs probability of a label being true according to logistic model.
def logistic_predictions(weights, inputs):
    return sigmoid(np.dot(inputs, weights))

# Training loss is the negative log-likelihood of the training labels.
def loss(weights, inputs, targets):
    preds = logistic_predictions(weights, inputs)
    label_logprobs = np.log(preds) * targets + np.log(1 - preds) * (1 - targets)
    return -np.sum(label_logprobs)

# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])

D = 3
N = 5
weights_true = onp.random.randn(D)
inputs = onp.random.randn(N, D)
logits = inputs.dot(weights_true)
targets = (onp.random.rand(N) < sigmoid(logits))


# Define a compiled function that returns gradients of the training loss
training_gradient_fun = jit(grad(loss))

# Optimize weights using gradient descent.
weights = onp.random.randn(D)
print("Initial loss: {:0.2f}".format(loss(weights, inputs, targets)))
lr = 0.1
max_iter = 100
for i in range(max_iter):
    weights -= lr * training_gradient_fun(weights, inputs, targets)

print("Trained loss: {:0.2f}".format(loss(weights, inputs, targets)))
