# optimzization utilties

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import inspect

def plot_loss_trace(losses, loss_min=None, ax=None):
    '''Plot loss vs number of function evals.
    losses is a list of floats'''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(losses[1:], '-')
    if loss_min is not None:
        ax.axhline(loss_min, 0, len(losses), color='r')
        # Make sure horizontal line is visible by changing yscale
        ylim = ax.get_ylim()
        ax.set_ylim([0.9*loss_min, 1.1*ylim[1]])
    return ax
    


class OptimLogger(object):
    '''Class to create a stateful callback function for optimizers,
    of the form callback(params).
    This calls eval_fun(params) whenever iter is a multiple of eval_freq,
    which can be used to evaluate validation performance.'''
    def __init__(self, eval_fun=None, eval_freq=0, store_freq=0, print_freq=0):
        self.param_trace = []
        self.eval_trace = []
        self.iter_trace = []
        self.eval_freq = eval_freq
        self.store_freq = store_freq
        self.print_freq = print_freq
        self.eval_fun = eval_fun
        self.iter = 0
        
    def callback(self, params):
        if (self.eval_freq > 0) and (self.iter % self.eval_freq == 0):
            obj = self.eval_fun(params)
            self.eval_trace.append(obj)
            self.iter_trace.append(self.iter)
            if self.print_freq > 0:
                print "iteration {}, objective {:2.3f}".format(self.iter, obj)
        if (self.store_freq > 0) and (self.iter % self.store_freq == 0):
            self.param_trace.append(np.copy(params)) 
        self.iter += 1

           

# Shuffle rows (for SGD)
def shuffle_data(X, y):
    N = y.shape[0]
    perm = np.arange(N)
    np.random.shuffle(perm)
    return X[perm], y[perm]
    
######
# Learning rate functions
    
def const_lr(lr=0.001):
    fn = lambda(iter): lr
    return fn
    
#https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#exponential_decay
#decayed_learning_rate = learning_rate *
#                        decay_rate ^ (global_step / decay_steps)
def lr_exp_decay(t, base_lr=0.001, decay_rate=0.9, decay_steps=100, staircase=True):
    '''To emulate a fixed learning rate, set decay_rate=0, decay_steps=inf'''
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
    for iter in range(500):
        lr = lr_exp_decay(iter, 0.01, 0, np.inf, True)
        lr_trace.append(lr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lr_trace)
    plt.show()

def grid_search_1d(eval_fun, param_list):
    #scores = np.apply_along_axis(eval_fun, 0, param_list)
    scores = [eval_fun(p) for p in param_list]
    istar = np.nanargmin(scores)
    return param_list[istar], scores

def lr_tuner(obj_fun, search_method, optimizer, grad_fun, x0,  max_iters, lrs=[1e-4,1]):
    def lr_eval_fun(lr):
        params, score = optimizer(obj_fun, grad_fun, x0, max_iters, None, const_lr(lr))
        print 'lr eval fun using {} gives {}'.format(lr, score)
        return score
    if search_method == 'grid':
        if len(lrs) == 2:
            lrs = np.logspace(np.log10(lrs[0]), np.log10(lrs[1]), 5)
        lr, scores = grid_search_1d(lr_eval_fun, lrs)
    else:
        res = opt.minimize_scalar(lr_eval_fun, bounds=[lrs[0], lrs[1]], method='bounded', \
                options = {'maxiter': 10, 'xatol': 1e-2, 'disp': False})
        scores = []
        lrs = []
        lr = res.x
    return lr, lrs, scores

######
# Minibatch functions
# From https://github.com/HIPS/neural-fingerprint/blob/2003a28d5ae4a78d99fdc06db8671b994f88c5a6/neuralfingerprint/util.py#L126-L138
def get_ith_minibatch_ixs(i, num_datapoints, batch_size):
    num_minibatches = num_datapoints / batch_size + ((num_datapoints % batch_size) > 0)
    i = i % num_minibatches
    start = i * batch_size
    stop = start + batch_size
    return slice(start, stop)

def build_batched_grad(grad, batch_size, inputs, targets):
    '''Grad has signature(weights, inputs, targets, N).
    Returns batched_grad with signature (weights, iter), applied to a
    minibatch. We pass in the overall dataset size, N, to act as 
    scaling factor.'''
    N = inputs.shape[0]
    def batched_grad(weights, i):
        cur_idxs = get_ith_minibatch_ixs(i, len(targets), batch_size)
        return grad(weights, inputs[cur_idxs], targets[cur_idxs], N)
    return batched_grad
    
######
# Modified from https://github.com/HIPS/autograd/blob/master/examples/optimizers.py

def maybe_add_iter_arg_to_fun(fun):
    '''This modifies a batch function to work in the online setting,
    by accepting (but ignoring) an iteration argument.'''
    if len(inspect.getargspec(fun)[0])==1:
        return lambda params, iter: fun(params)
    else:
        return fun
            
def autosgd(tuning_fun, tuning_method, obj_fun, grad_fun, x0, max_iters=100,
            callback=None, mass=0.9, update='regular'):
    lr, lrs, scores = lr_tuner(tuning_fun, tuning_method, sgd, grad_fun, x0, max_iters)
    print 'auto_sgd picked {} from {} with scores {}'.format(lr, lrs, scores)
    lr_fun = lambda iter: lr_exp_decay(iter, lr)
    x, val = sgd(obj_fun, grad_fun, x0, max_iters, callback, lr_fun, mass, update)
    return x, val, lr

def sgd(obj_fun, grad_fun, x0, max_iters=100, callback=None,
        lr_fun=const_lr(0.01), mass=0.9, update='regular', avgdecay=0.99):
    '''Stochastic gradient descent with momentum.
    See eg http://caffe.berkeleyvision.org/tutorial/solver.html'''
    x = np.copy(x0)
    xavg = x
    if callback is not None: callback(x)
    velocity = np.zeros(len(x))
    grad_fun = maybe_add_iter_arg_to_fun(grad_fun)
    for i in range(max_iters):
        lr = lr_fun(i)
        if update == 'regular': # standard momentum
            g = grad_fun(x, i)
            velocity = mass * velocity - lr * g
        elif update == 'nesterov': # nesterov accelerated gradient
            gg = grad_fun(x + mass * velocity, i)
            velocity = mass * velocity - lr * gg
        elif update == 'convex': # convex combination (autograd code)
            g = grad_fun(x, i)
            velocity = lr * (mass * velocity - (1.0 - mass) * g)
        else:
            raise ValueError('unknown update {}'.format(update))
        x  = x + velocity
        if callback is not None: callback(x)
    val = obj_fun(x)
    val_avg = obj_fun(xavg)
    print 'sgd: val {:0.4g}, val_avg {:0.4g}'.format(val, val_avg)
    if val < val_avg:
        return x, val
    else:
        return xavg, val_avg


def autoadam(tuning_fun, tuning_method, obj_fun, grad_fun, x0, max_iters=100,
            callback=None, b1=0.9, b2=0.999, eps=10**-8):
    lr, lrs, scores = lr_tuner(tuning_fun, tuning_method, adam, grad_fun, x0, max_iters)
    print 'auto_adam picked {} from {} with scores {}'.format(lr, lrs, scores)
    lr_fun = lambda iter: lr_exp_decay(iter, lr)
    x, val = adam(obj_fun, grad_fun, x0, max_iters, callback, lr_fun, b1, b2, eps)
    return x, val, lr

def adam(obj_fun, grad_fun, x0, max_iters=100, callback=None,
         lr_fun=const_lr(0.01), b1=0.9, b2=0.999, eps=10**-8, avgdecay=0.99):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    x = np.copy(x0)
    xavg = x
    if callback is not None: callback(x)
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    grad_fun = maybe_add_iter_arg_to_fun(grad_fun)
    for i in range(max_iters):
        g = grad_fun(x, i)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        step_size = lr_fun(i)
        x -= step_size*mhat/(np.sqrt(vhat) + eps)
        if callback: callback(x)
        # Polyak iterate averaging
        xavg = (1-avgdecay)*x + avgdecay*xavg
    val = obj_fun(x)
    val_avg = obj_fun(xavg)
    print 'adam: val {:0.4g}, val_avg {:0.4g}'.format(val, val_avg)
    if val < val_avg:
        return x, val
    else:
        return xavg, val_avg
        
def rmsprop(obj_fun, grad_fun, x0, max_iters=100,  callback=None,
            lr_fun=const_lr(0.01), gamma=0.9, eps = 10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    x = np.copy(x0)
    avg_sq_grad = np.ones(len(x))
    grad_fun = maybe_add_iter_arg_to_fun(grad_fun)
    if callback is not None: callback(x)
    for i in range(max_iters):
        g = grad_fun(x, i)
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        step_size = lr_fun(i)
        x -= step_size * g/(np.sqrt(avg_sq_grad) + eps)
        if callback: callback(x)
    val = obj_fun(x)
    return x, val



def bfgs(obj_fun, grad_fun, params, max_iters=100, callback_fun=None):
    '''This wraps scipy.minimize. So callback has the signature 
    callback(params).'''
    result = opt.minimize(obj_fun, params,  method='BFGS', jac=grad_fun,
            callback=callback_fun, options = {'maxiter':max_iters, 'disp':True, 'gtol': 1e-3})
    return result.x, result.fun
    
"""   
# Modified from 
# https://github.com/HIPS/neural-fingerprint/blob/2003a28d5ae4a78d99fdc06db8671b994f88c5a6/neuralfingerprint/optimizers.py
def bfgs_hips(obj_and_grad, x,  num_iters=100, callback=None):
    def epoch_counter():
        epoch = 0
        while True:
            yield epoch
            epoch += 1
    ec = epoch_counter()
    wrapped_callback=None
    if callback:
        def wrapped_callback(params):
            res = obj_and_grad(params)
            grad = res[1]
            callback(params, next(ec), grad)
    res =  opt.minimize(fun=obj_and_grad, x0=x, jac =True, callback=wrapped_callback,
                    method = 'BFGS', options = {'maxiter':num_iters, 'disp':True, 'gtol': 1e-3})
    return res.x
"""  

def branin(x):
    """Branin function.
    
    This function is widely used to evaluate nonconvex optimization methods.
    It is typically evaluated over the range -5 <= x1 <= 10, 0 <= x2 <= 15.
    
    This function has 3 global minima, at 
    x_global_min = [np.pi, 2.275];
    x_global_min = [-np.pi, 12.275]
    x_global_min = [9.42478, 2.475]
    The objective function at these points has value 0.397887
    
    Args:
        x: N*2 array of points (N = num. points to evaluate)
    
    Returns:
        f: N*1 function values at each x
        df: N*2 gradient vector at each x
    """
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8 * np.pi)
    x1 = x[:, 0]
    x2 = x[:, 1]
    z = x2 - b * np.square(x1) + c * x1 - r
    f = a * np.square(z) + s * (1 - t) * np.cos(x1) + 10
    df0 = 2 * a * np.inner(-2 * b * x1 + c, z) - s * (1-t) * np.sin(x1)
    df1 = 2 * a * z
    df = np.array([df0, df1]) 
    return f, df
    
######    
def main():
    plot_lr_trace()
    
if __name__ == "__main__":
    main()
    
