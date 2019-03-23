#!/usr/bin/env python

# Plots various neural net activation functions.

import matplotlib.pyplot as plt
import numpy as np
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")


e = np.exp(1)
x = np.linspace(-4, 4, 1000)
y = e**x / (e**x + 1)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('sigmoid function')
plt.savefig(os.path.join(figdir, 'sigmoidPlot.pdf'))
plt.show()

y = np.tanh(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('tanh function')
plt.savefig(os.path.join(figdir, 'tanhPlot.pdf'))
plt.show()


y = np.log(1+np.exp(x))
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('softplus function')
plt.savefig(os.path.join(figdir, 'softplusPlot.pdf'))
plt.show()


y = 1.0*(x>0)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('Heaviside function')
plt.savefig(os.path.join(figdir, 'heavisidePlot.pdf'))
plt.show()

y = np.maximum(0, x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('ReLU function')
plt.savefig(os.path.join(figdir, 'reluPlot.pdf'))
plt.show()

lam = 0.5
y = np.maximum(0, x) + lam*np.minimum(0, x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('Leaky ReLU function')
plt.savefig(os.path.join(figdir, 'leakyReluPlot.pdf'))
plt.show()

lam = 0.5
y = np.maximum(0, x) + np.minimum(0, lam*(e**x - 1))
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('ELU function')
plt.savefig(os.path.join(figdir, 'eluPlot.pdf'))
plt.show()

