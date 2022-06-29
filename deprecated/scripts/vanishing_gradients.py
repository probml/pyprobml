# Vanishing gradients for certain activation functions

# Based on 
#https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b

import superimport

import numpy as np
import matplotlib.pyplot as plt
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    p = sigmoid(x)
    return p*(1-p)

def relu(x):
    return np.maximum(0, x)

def heaviside(x):
    return (x > 0)

def relu_grad(x):
    return heaviside(x)

x = np.linspace(-10, 10, 100)
y = sigmoid(x);
plt.figure()
plt.plot(x, y)
plt.title('sigmoid function')
plt.savefig('../figures/sigmoid.pdf')
plt.show()

y = sigmoid_grad(x);
plt.figure()
plt.plot(x, y)
plt.title('derivative of sigmoid')
plt.ylim(0,1)
plt.savefig('../figures/sigmoid_deriv.pdf')
plt.show()

x = np.linspace(-10, 10, 100)
y = relu(x);
plt.figure()
plt.plot(x, y)
plt.title('relu function')
plt.savefig('../figures/relu.pdf')
plt.show()

y = relu_grad(x);
plt.figure()
plt.plot(x, y)
plt.title('derivative of relu')
plt.ylim(-0.1,1.1)
plt.savefig('../figures/relu_deriv.pdf')
plt.show()