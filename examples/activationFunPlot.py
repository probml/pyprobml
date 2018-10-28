#!/usr/bin/env python

# Plots various neural net activation functions.

import matplotlib.pyplot as plt
import numpy as np
import os

folder = "/Users/kpmurphy/github/pyprobml/figures"

e = np.exp(1)
x = np.linspace(-4, 4, 1000)
y = e**x / (e**x + 1)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('sigmoid function')
plt.show()
plt.savefig(os.path.join(folder, 'sigmoidPlot.pdf'))

y = np.tanh(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('tanh function')
plt.show()
plt.savefig(os.path.join(folder, 'tanhPlot.pdf'))

y = np.log(1+np.exp(x))
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('softplus function')
plt.show()
plt.savefig(os.path.join(folder, 'softplusPlot.pdf'))


y = 1.0*(x>0)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('Heaviside function')
plt.show()
plt.savefig(os.path.join(folder, 'heavisidePlot.pdf'))

y = np.maximum(0, x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('ReLU function')
plt.show()
plt.savefig(os.path.join(folder, 'reluPlot.pdf'))

lam = 0.5
y = np.maximum(0, x) + lam*np.minimum(0, x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('Leaky ReLU function')
plt.show()
plt.savefig(os.path.join(folder, 'leakyReluPlot.pdf'))

lam = 0.5
y = np.maximum(0, x) + np.minimum(0, lam*(e**x - 1))
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('ELU function')
plt.show()
plt.savefig(os.path.join(folder, 'eluPlot.pdf'))

