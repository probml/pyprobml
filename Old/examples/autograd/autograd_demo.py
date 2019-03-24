#!/usr/bin/env python

# Example from https://github.com/HIPS/autograd.

import autograd
import autograd.numpy as np  # Thinly-wrapped numpy
import matplotlib.pyplot as plt

def tanh(x):                 # Define a function
    y = np.exp(-x)
    return (1.0 - y) / (1.0 + y)

grad_tanh = autograd.grad(tanh)       # Obtain its gradient function
g_auto = grad_tanh(1.0)               # Evaluate the gradient at x = 1.0
g_finite = (tanh(1.0001) - tanh(0.9999)) / 0.0002  # Compare to finite differences
assert(np.allclose(g_auto, g_finite))

def elementwise_grad(fun):                   # A wrapper for broadcasting
    return autograd.grad(lambda x: np.sum(fun(x)))    # (closures are no problem)

grad_tanh   = elementwise_grad(tanh)
grad_tanh_2 = elementwise_grad(grad_tanh)
x = np.linspace(-7, 7, 200)

plt.plot(x, tanh(x), 'r-')
plt.plot(x, grad_tanh(x), 'k-')
plt.plot(x, grad_tanh_2(x), 'b-')
plt.show()
