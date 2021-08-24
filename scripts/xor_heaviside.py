# Show that 2 layer MLP (with manually chosen weights) can solve the XOR problem
# Based on
# https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb

import superimport

import numpy as np
import matplotlib.pyplot as plt

import pyprobml_utils as pml


def heaviside(z):
    return (z >= 0).astype(z.dtype)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def mlp_xor(x1, x2, activation=heaviside):
    return activation(-activation(x1 + x2 - 1.5) + activation(x1 + x2 - 0.5) - 0.5)

x1s = np.linspace(-0.2, 1.2, 100)
x2s = np.linspace(-0.2, 1.2, 100)
x1, x2 = np.meshgrid(x1s, x2s)

z1 = mlp_xor(x1, x2, activation=heaviside)
z2 = mlp_xor(x1, x2, activation=sigmoid)

#plt.figure(figsize=(10,4))
plt.figure()

#plt.subplot(121)
plt.contourf(x1, x2, z1)
plt.plot([0, 1], [0, 1], "gs", markersize=20)
plt.plot([0, 1], [1, 0], "r^", markersize=20)
plt.title("Activation function: heaviside", fontsize=14)
plt.grid(True)
pml.save_fig("xor-heaviside.pdf")
plt.show()

plt.figure()
#plt.subplot(122)
plt.contourf(x1, x2, z2)
plt.plot([0, 1], [0, 1], "gs", markersize=20)
plt.plot([0, 1], [1, 0], "r^", markersize=20)
plt.title("Activation function: sigmoid", fontsize=14)
plt.grid(True)
pml.save_fig("xor-sigmoid.pdf")
plt.show()

