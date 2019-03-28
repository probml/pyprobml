# Plots various neural net activation functions.

import matplotlib.pyplot as plt
import numpy as np
from pyprobml_utils import save_fig

e = np.exp(1)
x = np.linspace(-4, 4, 1000)
y = e**x / (e**x + 1)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('sigmoid function')
save_fig('sigmoidPlot.pdf')
plt.show()

y = np.tanh(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('tanh function')
save_fig('tanhPlot.pdf')
plt.show()


y = np.log(1+np.exp(x))
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('softplus function')
save_fig('softplusPlot.pdf')
plt.show()


y = 1.0*(x>0)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('Heaviside function')
save_fig('heavisidePlot.pdf')
plt.show()

y = np.maximum(0, x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('ReLU function')
save_fig('reluPlot.pdf')
plt.show()

lam = 0.5
y = np.maximum(0, x) + lam*np.minimum(0, x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('Leaky ReLU function')
save_fig('leakyReluPlot.pdf')
plt.show()

lam = 0.5
y = np.maximum(0, x) + np.minimum(0, lam*(e**x - 1))
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('ELU function')
save_fig('eluPlot.pdf')
plt.show()

