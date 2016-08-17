# Optimization of the 2d Rosen function
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import os
import utils.util as util
import utils.optim as optim
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy import optimize as opt
import scipy.linalg as la

#http://people.duke.edu/~ccc14/sta-663/BlackBoxOptimization.html

def reporter(p):
    """Reporter function to capture intermediate states of optimization."""
    global ps
    #ps.append(p)
    ps.append(np.copy(p))

def plot_trace(ps, ttl):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosen(np.vstack([X.ravel(), Y.ravel()])).reshape((100,100))
    ps = np.array(ps)
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.contour(X, Y, Z, np.arange(10)**5)
    plt.plot(ps[:, 0], ps[:, 1], '-o')
    plt.plot(1, 1, 'r*', markersize=12) # global minimum
    plt.subplot(122)
    plt.semilogy(range(len(ps)), rosen(ps.T))
    plt.title(ttl)

# Initial starting position
x0 = np.array([4,-4.1])

#logger = optim.OptimLogger(rosen, 1, 1, 1)

ps = [x0]
res = opt.minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hess=rosen_hess, callback=reporter)
print '\nresults of Newton-CG'
print res
plot_trace(ps, 'Newton, obj {:0.4f}'.format(res.fun))

ps = [x0]
res = opt.minimize(rosen, x0, method='BFGS', jac=rosen_der, callback=reporter)
print '\nresults of BFGS'
print res
plot_trace(ps, 'BFGS, obj {:0.4f}'.format(res.fun))

ps = [x0]
lr = 0.0001
params, obj = optim.sgd(rosen, rosen_der, x0, 100, reporter, optim.const_lr(lr), 0)
print '\nGD final params {}'.format(params)
plot_trace(ps, 'GD({:0.4f}), obj {:0.4f}'.format(lr, obj))

ps = [x0]
lr = 0.0002
params, obj = optim.sgd(rosen, rosen_der, x0, 100, reporter, optim.const_lr(lr), 0)
print '\nGD final params {}'.format(params)
plot_trace(ps, 'GD({:0.4f}), obj {:0.4f}'.format(lr, obj))

ps = [x0]
lr = 0.0004
params, obj = optim.sgd(rosen, rosen_der, x0, 100, reporter, optim.const_lr(lr), 0)
print '\nGD final params {}'.format(params)
plot_trace(ps, 'GD({:0.4f}), obj {:0.4f}'.format(lr, obj))


ps = [x0]
params, obj, lr = optim.autoadam(rosen, 'bisection', rosen, rosen_der, x0, 100, reporter)
print '\nAdam final params {}'.format(params)
plot_trace(ps, 'Adam(Bisection {:0.4f}), obj {:0.4f}'.format(lr, obj))


plt.show()
