# 1d linear regression.
# Make the data and plot it.

import matplotlib.pyplot as plt
import numpy as np
from utils.linreg_model import LinregModel

def make_fun_1d(fun_type):
    if fun_type == 'linear':
        params_true = np.array([1, 1])
        true_fun = lambda x: params_true[1] + params_true[0]*x
        ttl = 'w*x + b'
    if fun_type == 'sine':
        params_true = np.array([1, 1])
        true_fun = lambda x: params_true[1] + params_true[0]*np.sin(x)
        ttl = 'w*sin(x) + b' 
    if fun_type == 'quad':
        # 'thibaux' parameters from here:
        #https://github.com/probml/pmtk3/blob/master/matlabTools/stats/polyDataMake.m
         #w = [-1.5; 1/9];
        #fun = @(x) w(1)*x + w(2)*x.^2;
        params_true = np.array([1, 1])
        true_fun = lambda x: params_true[0]*x + params_true[1]*np.square(x)
        ttl = 'w*x^2 + b' 
    return params_true, true_fun, ttl
        
# Sample (x,y) pairs from a noisy function 
def make_data_linreg_1d(N, fun_type, matlab_hack=False):
    if '-' in fun_type:
        parts = fun_type.split('-')
        fun_type = parts[0]
        centered = (parts[1] == 'centered')
    else:
        centered = True
    params_true, true_fun, fun_name = make_fun_1d(fun_type) 
    if centered:
        xtrain = np.random.normal(0, 3, N)
    else:
        xtrain = np.linspace(0, 20, N)
    ytrain_clean = true_fun(xtrain) 
    sigma2 = 1
    noise = np.random.normal(0, 1, xtrain.shape) * np.sqrt(sigma2)
    Ytrain = ytrain_clean + noise   
    Xtrain = np.reshape(xtrain, (N, 1))
    if matlab_hack:
        # Match matlab's https://github.com/probml/pmtk3/blob/master/demos/contoursSSEdemo.m
        Ytrain = np.array([3.560, 1.831, 0.924, 1.475, 0.531, 0.686, 2.720, 3.061, -1.233, 1.369, 2.990, 1.546, 3.425,
        1.406, 3.306, 0.729, 4.050, 1.879, 0.313, 0.369])
        xtrain = np.array([1.749, 0.133, 0.325, -0.794, 0.315, -0.527, 0.932, 1.165, -2.046, -0.644, 1.741, 0.487, 1.049, 
        1.489, 1.271, -1.856, 2.134, 1.436, -0.917, -1.106])
        N = len(xtrain)
        Xtrain = np.reshape(xtrain, [N,1])
    return Xtrain, Ytrain, params_true, true_fun, fun_name

# Plot 2d error surface around model's parameter vaulues
def plot_error_surface_2d(loss_fun, params_opt, params_true, fun_type, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111) 
    # Choose the range to make a pretty plot
    if fun_type == 'quad':
        w0_range = [-1, 2] 
        w1_range = [-1, 6] 
    elif fun_type == 'sine':
        w0_range = [-3, 3] 
        w1_range = [-1, 3] 
    else:
        w0_range = [-1, 3] 
        w1_range = [-1, 3] 
    w0s = np.linspace(w0_range[0], w0_range[1], 100)
    w1s = np.linspace(w1_range[0], w1_range[1], 100)
    w0_grid, w1_grid = np.meshgrid(w0s, w1s)
    lossvec = np.vectorize(loss_fun)
    z = lossvec(w1_grid, w0_grid)
    cs = ax.contour(w1s, w0s, z)
    ax.clabel(cs)
    ax.plot(params_opt[1], params_opt[0], 'rx', markersize=14)
    ax.plot(params_true[1], params_true[0], 'k+', markersize=14)
    return ax

def plot_param_trace_2d(params_trace, ax):
    '''Plot 2d trajectory of parameters on top of axis,
    param_trace is list of weight vectors'''
    n_steps = len(params_trace)
    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)
    for step in range(1, n_steps):
        xs[step] = params_trace[step][1]
        ys[step] = params_trace[step][0]
    ax.plot(xs, ys, 'o-')
    
def plot_data_and_predictions_1d(xtrain, ytrain, true_fun, pred_fun, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(xtrain, ytrain, 'o', label='observed')
    x_range = np.linspace(np.min(xtrain), np.max(xtrain), 100)
    x_range = np.reshape(x_range, [100,1])
    yhat_range = pred_fun(x_range)
    ax.plot(x_range, yhat_range, 'r--', label='predicted')
    y_range = true_fun(x_range)
    ax.plot(x_range, y_range, 'k-', label='truth')
    return ax
    
def main():
    for fun_type in ['linear', 'quad', 'sine']:
        np.random.seed(1)
        N = 20
        Xtrain, Ytrain, params_true, true_fun, ttl = make_data_linreg_1d(N, fun_type)
    
        model = LinregModel(1, True)
        params_ols, loss_ols = model.ols_fit(Xtrain, Ytrain)
        print(ttl)
        print params_ols
        
        # Plot data
        predict_fun = lambda x: model.prediction(params_ols, x)
        ax = plot_data_and_predictions_1d(Xtrain, Ytrain, true_fun, predict_fun)
        ax.set_title(ttl)
        
        # Plot error surface
        loss_fun = lambda w0, w1: model.objective([w0, w1], Xtrain, Ytrain)
        ax  = plot_error_surface_2d(loss_fun, params_ols, params_true, fun_type)
        ax.set_title(ttl)
    plt.show()
    
if __name__ == "__main__":
    main()
    
