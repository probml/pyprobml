# Functions related to stochastic gradient descent (SGD)
# DEPRECATED
# Use optim_util.py instead

import numpy as np
import matplotlib.pyplot as plt
#from collections import namedtuple
from scipy.optimize import minimize

# Class to create a stateful callback function for sgd
class SGDLogger(object):
    def __init__(self, print_freq=0, store_params=False):
        self.param_trace = []
        self.grad_norm_trace = []
        self.obj_trace = []
        self.iter = 0
        self.print_freq = print_freq
        self.store_params = store_params
        
    def update(self, params, obj, gradient, epoch, batch_num):
        self.obj_trace.append(obj)
        self.grad_norm_trace.append(np.linalg.norm(gradient))
        if self.store_params:
            self.param_trace.append(params) 
        if (self.print_freq > 0) and (self.iter % self.print_freq == 0):
            print "epoch {}, batch num {}, iteration {}, objective {:2.3f}".format(
                epoch, batch_num, self.iter, obj)
        self.iter += 1
    
class MinimizeLogger(object):
    '''Class to create a stateful callback function for scipy's minimize'''
    def __init__(self, obj_fun, grad_fun, args, print_freq=0, store_params=False):
        self.args = args
        self.param_trace = []
        self.obj_trace = []
        self.iter = 0
        self.print_freq = 0
        self.obj_fun = obj_fun
        self.grad_fun = grad_fun
        self.grad_norm_trace = []
        self.store_params = store_params
        
    def update(self, params):
        obj = self.obj_fun(params, *self.args)
        self.obj_trace.append(obj)
        gradient = self.grad_fun(params, *self.args)
        self.grad_norm_trace.append(np.linalg.norm(gradient))
        if self.store_params:
            self.param_trace.append(params) 
        if (self.print_freq > 0) and (self.iter % self.print_freq == 0):
            print "iteration {}, objective {:2.3f}".format(self.iter, obj)
        self.iter += 1


def bfgs_fit(params, obj_fun, grad_fun, args=None, callback_fun=None, num_iters=100):
    result = minimize(obj_fun, params, args, method='BFGS', jac=grad_fun,
            callback=callback_fun,
            options = {'maxiter':num_iters, 'disp':True})
    print 'finished bfgs_fit after {} iterations, {} fun evals, {} grad evals'.format(result.nit, result.nfev, result.njev)
    #n_fun_evals = result.nfev + result.njev # function plus gradient
    n_fun_evals = result.nfev 
    return result.x, result.fun, n_fun_evals

    
#########
# SGD helpers

# Shuffle rows (for SGD)
def shuffle_data(X, y):
    N = y.shape[0]
    perm = np.arange(N)
    np.random.shuffle(perm)
    return X[perm], y[perm]

def make_batches(N_data, batch_size):
    return [slice(i, min(i+batch_size, N_data))
            for i in range(0, N_data, batch_size)]
                        
# Batchifier class based on
#https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/mnist/input_data.py

class MiniBatcher(object):
    def __init__(self, X, y, batch_size):
        self.num_examples = X.shape[0]
        self.batch_size = batch_size
        self.X, self.y = shuffle_data(X, y) 
        #self.X, self.y = X, y
        self.index_in_epoch = 0
        self.epochs_completed = 0
        
    def next_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        if self.index_in_epoch > self.num_examples:
            self.epochs_completed += 1
            self.X, self.y = shuffle_data(self.X, self.y)
            start = 0
            self.index_in_epoch = self.batch_size
        stop = self.index_in_epoch
        return self.X[start:stop], self.y[start:stop]

# simple test
if False:
    np.random.seed(1)
    batch_size = 5
    Xtrain = np.arange(0,100)
    ytrain = Xtrain
    batchifier = MiniBatcher(Xtrain, ytrain, batch_size)
    for iter in range(5):
        Xb, yb = batchifier.next_batch()
        print iter
        print Xb




######
# Learning rate functions
    
#https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#exponential_decay
#decayed_learning_rate = learning_rate *
#                        decay_rate ^ (global_step / decay_steps)
def lr_exp_decay(t, base_lr=0.001, decay_rate=0.9, decay_steps=2, staircase=True):
    if staircase:
        exponent = t / decay_steps # integer division
    else:
        exponent = t / np.float(decay_steps) 
    return base_lr * np.power(decay_rate, exponent)
   
   
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html                                                                            
# eta = eta0 / pow(t, power_t) [default]
def lr_inv_scaling(t, base_lr=0.001, power_t=0.25):
   return base_lr / np.power(t+1, power_t)
   

#http://leon.bottou.org/projects/sgd
def lr_bottou(t, base_lr=0.001, power_t=0.75, lam=1):
   return base_lr / np.power(1 + lam*base_lr*t, power_t)


def plot_lr_trace():
    lr_trace = []
    for iter in range(10):
        #lr = lr_inv_scaling(iter)
        lr = lr_exp_decay(iter, 0.01, 1)
        lr_trace.append(lr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lr_trace)
    plt.show()
    return(lr_trace)


#######
# Summary of these algorithms can be found here
#http://sebastianruder.com/optimizing-gradient-descent/index.html

def sgd_minimize(params, obj_fun, grad_fun, X, y, batch_size, num_epochs, 
       param_updater, callback_fun=None): 
    param_updater.init(params)                                                                                                                                                                                                
    N = X.shape[0]
    batch_indices = make_batches(N, batch_size)
    D = len(params)
    params_avg = np.zeros(D)
    decay = 0.99
    iter = 0
    for epoch in range(num_epochs):
        X, y = shuffle_data(X, y) 
        for batch_num, batch_idx in enumerate(batch_indices):
            iter = iter + 1
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            obj_value = obj_fun(params, X_batch, y_batch)
            gradient_vec = grad_fun(params, X_batch, y_batch)
            params = param_updater.update(params, obj_value, gradient_vec, epoch)
            params_avg = decay * params_avg + (1-decay) * params
            params_avg_unbiased = params_avg / (1 - (decay**iter))
            if callback_fun is not None:
                callback_fun(params, obj_value, gradient_vec, epoch, batch_num) 
    print 'finished sgd, after {} minibatch updates'.format(iter) 
    loss = obj_fun(params, X, y)
    loss_avg = obj_fun(params_avg, X, y)
    return params, loss, iter, params_avg, loss_avg

##########

  
class Rprop(object):
    def __init__(self, eta_plus = 1.2, eta_minus = 0.5, delta_min = 0, delta_max = 50, delta_zero = 0.5, improved_Rprop=True):
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta_init = delta_zero
        self.improved_Rprop = improved_Rprop
        self.iter = 0

    def init(self, params):
        D = len(params)
        self.old_delta = self.delta_init * np.ones(D)
        self.old_gradient = np.zeros(D)
        self.old_delta_params = np.zeros(D)
        self.old_objective = 0
        
    def update(self, params, obj_value, grad, epoch):
        if self.iter == 0:
            ip = np.ones(len(params))
            delta = self.delta_init
        else:
            ip = self.old_gradient * grad
            delta = np.minimum(self.old_delta * self.eta_plus, self.delta_max) * (ip > 0) + \
                np.maximum(self.old_delta * self.eta_minus, self.delta_min) * (ip < 0) + \
                self.old_delta * (ip == 0)
        delta_params = np.zeros(len(params))
        delta_params = -np.sign(grad) * delta * (ip > 0) + -np.sign(grad) * delta * (ip == 0) 
        if self.improved_Rprop:
            if (obj_value > self.old_objective): 
                delta_params[ip < 0] = -self.old_delta_params[ip < 0] # conditionally backtrack
        else:
            delta_params[ip < 0] = -self.old_delta_params[ip < 0] # backtrack
        params = params + delta_params
        self.old_delta = delta
        self.old_gradient = grad
        self.old_gradient[ip < 0] = 0 # just changed sign, reset counter to 0 
        self.old_objective = obj_value
        self.old_delta_params = delta_params
        return params

        
class SGDMomentum(object):
    def __init__(self, lr_fun, mass=0.9):
        self.lr_fun = lr_fun
        self.mass = mass # gamma
        self.velocity = []
        self.iter = 0
        
    def init(self, params):
        D = len(params)
        self.velocity = np.zeros(D)
        self.mean_squared_gradients = np.zeros(D)
        
    def update(self, params, obj_value, gradient_vec, epoch):
        self.iter = self.iter + 1
        lr = self.lr_fun(self.iter, epoch)
        self.velocity = self.mass * self.velocity + lr * gradient_vec 
        params = params - self.velocity
        return params

class RMSprop(object):
    def __init__(self, lr_fun, grad_sq_decay=0.9):
        self.lr_fun = lr_fun
        self.grad_sq_decay = grad_sq_decay
        self.mean_sq_gradients = []
        self.iter = 0
        
    def init(self, params):
        D = len(params)
        self.mean_sq_gradients = np.zeros(D)
        
    def update(self, params, obj_value, gradient_vec, epoch):
        self.iter = self.iter + 1
        lr = self.lr_fun(self.iter, epoch)
        self.mean_sq_gradients = self.grad_sq_decay * self.mean_sq_gradients + \
            (1-self.grad_sq_decay) * np.square(gradient_vec)
        eps = 1e-8
        denominator = np.sqrt(self.mean_sq_gradients) + eps
        params = params -  lr * gradient_vec / denominator
        return params

            
class ADAM(object):
    def __init__(self, lr_fun, grad_decay=0.9, grad_sq_decay=0.999):
        self.lr_fun = lr_fun # alpha
        self.grad_decay = grad_decay # beta1
        self.grad_sq_decay = grad_sq_decay # beta2
        self.mean_gradients = [] # m
        self.mean_sq_gradients = [] # v
        self.iter = 0
        
    def init(self, params):
        D = len(params)
        self.mean_gradients = np.zeros(D)
        self.mean_sq_gradients = np.zeros(D)
        
    def update(self, params, obj_value, gradient_vec, epoch):
        self.iter = self.iter + 1
        lr = self.lr_fun(self.iter, epoch)
        self.mean_gradients = self.grad_decay * self.mean_gradients + \
            (1-self.grad_decay) * gradient_vec
        self.mean_sq_gradients = self.grad_sq_decay * self.mean_sq_gradients + \
            (1-self.grad_sq_decay) * np.square(gradient_vec)
        eps = 1e-8
        mean_grad = self.mean_gradients / (1 - self.grad_decay**self.iter)
        mean_sq_grad = self.mean_sq_gradients / (1 - self.grad_sq_decay**self.iter)
        denominator = np.sqrt(mean_sq_grad) + eps
        params = params -  lr * mean_grad / denominator
        return params
