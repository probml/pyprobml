# Illustrate expoentnially-weighted moving average
# Based on 
#http://people.duke.edu/~ccc14/sta-663-2019/notebook/S09G_Gradient_Descent_Optimization.html#Smoothing-with-exponentially-weighted-averages
#http://people.duke.edu/~ccc14/sta-663-2018/notebooks/S09G_Gradient_Descent_Optimization.html

import superimport

import numpy as np
import matplotlib.pyplot as plt
from pyprobml_utils import save_fig

def ema(y, beta):
    """Exponentially weighted average."""
    n = len(y)
    zs = np.zeros(n)
    z = 0
    for i in range(n):
        z = beta*z + (1 - beta)*y[i]
        zs[i] = z
    return zs

def ema_debiased(y, beta):
    """Exponentially weighted average with hias correction."""
    n = len(y)
    zs = np.zeros(n)
    z = 0
    for i in range(n):
        z = beta*z + (1 - beta)*y[i]
        zc = z/(1 - beta**(i+1))
        zs[i] = zc
    return zs


np.random.seed(0)
n = 50
x = np.arange(n) * np.pi
y = np.cos(x) * np.exp(x/100) - 10*np.exp(-0.01*x)

betas = [0.9, 0.99]
for i, beta in enumerate(betas):
    plt.figure()
    plt.plot(x, y, 'o-')
    plt.plot(x, ema(y, beta), c='red', label='EMA')
    plt.plot(x, ema_debiased(y, beta), c='orange', label='EMA with bias correction')
    plt.title('beta = {:0.2f}'.format(beta))
    plt.legend()
    name = 'EMA{}.pdf'.format(i)
    save_fig(name)
    plt.show()
  