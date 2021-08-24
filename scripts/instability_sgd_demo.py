# Demo of gradient descent vs proximal methods on a quadratic objective.
# We show that GD can "blow up" if the initial step size is too large,
# but proximal methods are robust.

import superimport

import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt

import os
figdir = "../figures" # set this to '' if you don't want to save figures
def save_fig(fname):
    if figdir:
        plt.savefig(os.path.join(figdir, fname))


beta = 0.5
T0 = 5
T = 10
x0 = 0.5
lr_critical = 3 * T0**beta
lrs = [1.1*lr_critical, lr_critical, 0.5*lr_critical]

# Quadratic objective
# f(x)= 0.5 x*2

# Gradient descent
# x(t+1) = x(t) - lr*f'(x(t-1)) = (1-lr)*x(t-1)
def gd(x0, T, lr0, beta):
    xs = np.zeros(T)
    xs[0] = x0
    for t in range(1, T):
        lr = lr0/(t**beta)
        xs[t] = (1-lr)*xs[t-1]
    return xs

# Proximal point method
# x(t+1) = min_x [f(x)  + 1/2lr * (x-x(t))^2] = x(t)/(1+lr) 
def prox(x0, T, lr0, beta):
    xs = np.zeros(T)
    xs[0] = x0
    for t in range(1, T):
        lr = lr0/(t**beta)
        xs[t] = xs[t-1]/(1+lr)
    return xs

# Approximate proximal point, using truncated linear model of f.
# x(t+1) = min_x [ max{0,f(x(t)) + f'(x(t))*(x-x(t))}  + 1/2lr * (x-x(t))^2]
# = max{-0, (1-lr)*x(t)}
def trunc(x0, T, lr0, beta):
    xs = np.zeros(T)
    xs[0] = x0
    for t in range(1, T):
        lr = lr0/(t**beta)
        xs[t] = max(0, (1-lr)*xs[t-1])
    return xs
    

expts = {}
for i, lr0 in enumerate(lrs):
    x_trace = gd(x0, T, lr0, beta)
    name = 'GD-{:0.3f}'.format(lr0)
    expts[name] = x_trace
    
plt.figure()
for name in expts.keys():
    plt.plot(expts[name], label=name)
plt.legend()
save_fig('instability-gd.pdf')
plt.show()


expts = {}
for i, lr0 in enumerate(lrs):
    x_trace = prox(x0, T, lr0, beta)
    name = 'prox-{:0.3f}'.format(lr0)
    expts[name] = x_trace
    
plt.figure()
for name in expts.keys():
    plt.plot(expts[name], label=name)
plt.legend()
save_fig('instability-prox.pdf')
plt.show()


expts = {}
for i, lr0 in enumerate(lrs):
    x_trace = trunc(x0, T, lr0, beta)
    name = 'trunc-{:0.3f}'.format(lr0)
    expts[name] = x_trace
    
plt.figure()
for name in expts.keys():
    plt.plot(expts[name], label=name)
plt.legend()
save_fig('instability-trunc.pdf')
plt.show()


###############

'''
beta = 0.5
T0 = 5
T = 10
lrs = np.zeros(T)
lr0 = 3 * T0**beta
lrs[0] = lr0

# learning rate schedule
for t in range(1, T):
    lrs[t] = lr0/(t**beta)

# Gradient descent
xs = np.zeros(T)
xs[0] = 0.5 # random non-zero value
for t in range(1, T):
    xs[t] = (1-lrs[t])*xs[t-1]
    
plt.figure()
plt.plot(xs, label='GD')
plt.legend()
save_fig('instability-gd.pdf')
plt.show()

# Proximal point method
ys = np.zeros(T)
ys[0] = xs[0]
for t in range(1,T):
    ys[t] = ys[t-1]/(1+lrs[t])

plt.figure()
plt.plot(ys, label='Prox')
plt.legend()
save_fig('instability-prox.pdf')
plt.show()

# Aprox trunc
zs = np.zeros(T)
zs[0] = xs[0]
for t in range(1,T):
    zs[t] = max(0, (1-lrs[t])*zs[t-1])

plt.figure()
plt.plot(zs, label='Aprox-Trunc')
plt.legend()
save_fig('instability-trunc.pdf')
plt.show()

'''
