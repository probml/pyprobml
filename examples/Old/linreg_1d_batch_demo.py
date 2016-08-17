# 1d linear regression using batch optimization

import numpy as np
import matplotlib.pyplot as plt

from utils.optim_util import bfgs_fit, MinimizeLogger, plot_loss_trace
from demos.linreg_1d_plot_demo import plot_error_surface, make_data_linreg_1d
from utils.linreg_model import LinregModel

def plot_error_surface_and_param_trace(xtrain, ytrain, model, params_trace, ttl=None, ax=None):
    '''param_trace is list of weight vectors'''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    plot_error_surface(xtrain, ytrain, model, ax)
    n_steps = len(params_trace)
    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)
    for step in range(1, n_steps):
        xs[step] = params_trace[step][0]
        ys[step] = params_trace[step][1]
    ax.plot(xs, ys, 'o-')
    if ttl is not None:
        ax.set_title(ttl)


def main():
    np.random.seed(1)
    xtrain, ytrain, w_true = make_data_linreg_1d()
    N = xtrain.shape[0]
    D = 2
    Xtrain = np.c_[np.ones(N), xtrain] # add column of 1s
    
    w_init = np.zeros(D)
    model = LinregModel(w_init)
    logger = MinimizeLogger(model.objective, (Xtrain, ytrain), print_freq=10)
    
    params_ols, loss_ols = LinregModel.ols_fit(Xtrain, ytrain)
    params_bfgs, loss_bfgs, logger = bfgs_fit(Xtrain, ytrain, model, logger) 
    assert(np.allclose(params_bfgs, params_ols))
    assert(np.allclose(loss_bfgs, loss_ols))
   
    params_autograd, loss_autograd = bfgs_fit(Xtrain, ytrain, model,
            logger=None, use_autograd=True) 
    assert(np.allclose(params_bfgs, params_autograd))
    assert(np.allclose(loss_bfgs, loss_autograd))
    print "All assertions passed"

    print logger.obj_trace
    plot_loss_trace(logger.obj_trace, loss_ols, 'BFGS')  
    model_true = LinregModel(w_true)
    plot_error_surface_and_param_trace(xtrain, ytrain, model_true, logger.param_trace, 'BFGS')
    plt.show()
    
if __name__ == "__main__":
    main()
    