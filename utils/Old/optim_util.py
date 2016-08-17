# optimzization utilties

#import numpy as np
import autograd
import autograd.numpy as np  # Thinly-wrapped numpy
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def squared_loss(y_pred, y):
    '''y is N*1, y_pred is N*1.
    Returns scalar'''
    N = y.shape[0]
    return sum(np.square(y - y_pred))/N

# Function to plot loss vs time
def plot_loss_trace(losses, loss_min=None, ttl=None, ax=None):
    ''' losses is a list of floats'''
    training_steps = len(losses)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(range(0, training_steps), losses, 'o-')
    if loss_min is not None:
        ax.axhline(loss_min, 0, training_steps)
        # Make sure horizontal line is visible by changing yscale
        ylim = ax.get_ylim()
        ax.set_ylim([0.9*loss_min, ylim[1]])
    if ttl is not None:
       ax.set_title(ttl)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Training steps")
    
# Class to create a stateful callback function for scipy's minimize
class MinimizeLogger(object):
    def __init__(self, obj_fun, args, print_freq=0):
        self.args = args
        self.param_trace = []
        self.obj_trace = []
        self.iter = 0
        self.print_freq = 0
        self.obj_fun = obj_fun
        
    def update(self, params):
        obj = self.obj_fun(params, *self.args)
        self.obj_trace.append(obj)
        self.param_trace.append(params) # could take a lot of space
        if (self.print_freq > 0) and (iter % self.print_freq == 0):
            print "iteration {0}, objective {0:2.3f}".format(iter, obj)
        self.iter += 1

def bfgs_fit(Xtrain, ytrain, model, logger=None, use_autograd=False):
    if logger is None:
        callback_fun = None
    else:
        callback_fun = logger.update
    if use_autograd:
        grad_fun = autograd.grad(model.objective)
    else:
        grad_fun = model.gradient
    result = minimize(model.objective, model.params, (Xtrain, ytrain),
            method='BFGS', jac=grad_fun, callback=callback_fun)
    if logger is None:
        return result.x, result.fun
    else:
        return result.x, result.fun, logger
