# Plots various neural net activation functions.

import numpy as np
import matplotlib.pyplot as plt
import os
figdir = os.path.join(os.environ["PYPROBML"], "figures")
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def heaviside(z):
    return (z > 0)

z = np.linspace(-5, 5, 200)

plt.figure(figsize=(11,4))
plt.plot(z, heaviside(z), "r-", linewidth=1, label="Heaviside")
plt.plot(z, sigmoid(z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
plt.legend(loc="center right", fontsize=14)
plt.title("Activation functions", fontsize=14)
plt.axis([-5, 5, -1.2, 1.2])
save_fig('activationFuns.pdf')
plt.show()

'''
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
'''
