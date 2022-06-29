# Bayesian optimization of 1d continuous function
# Modified from Martin Krasser's code
# https://github.com/krasserm/bayesian-machine-learning/blob/dev/bayesian-optimization/bayesian_optimization.ipynb


import superimport

import numpy as np
from bayes_opt_utils import BayesianOptimizer, MultiRestartGradientOptimizer, expected_improvement

import matplotlib.pyplot as plt
import pyprobml_utils as pml
save_figures = True #False

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

np.random.seed(0)


def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    X = np.atleast_2d(X)
    #Y = np.atleast_2d(Y)
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     alpha=0.1) 
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X.ravel(), Y.ravel(), 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()    
        
def plot_convergence(X_sample, Y_sample, n_init=2):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x)+1)
    
    x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)
    
    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    
    
##################
    
bounds = np.array([[-1.0, 2.0]])
noise = 0.2

def f(X, noise=noise):
    return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)

X_init = np.array([[-0.9], [1.1]])
Y_init = f(X_init)


# Dense grid of points within bounds
X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

# Noise-free objective function values at X 
Y = f(X,0)

# Plot optimization objective with noise level 
plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
plt.plot(X, f(X), 'bx', lw=1, alpha=0.1, label='Noisy samples')
plt.plot(X_init, Y_init, 'kx', mew=3, label='Initial samples')
plt.legend()
if save_figures: pml.savefig('bayes-opt-init.pdf')
plt.show()


################


kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)


"""
https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/gaussian_process/kernels.py#L1287
The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.
"""


# Keep track of visited points for plotting purposes
global X_sample, Y_sample
X_sample = X_init
Y_sample = Y_init

def callback(X_next, Y_next, i):
  global X_sample, Y_sample
  # Plot samples, surrogate function, noise-free objective and next sampling location
  #plt.subplot(n_iter, 2, 2 * i + 1)
  plt.figure()
  plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i==0)
  plt.title(f'Iteration {i+1}')
  if save_figures: pml.savefig('bayes-opt-surrogate-{}.pdf'.format(i+1))
  plt.show()
  
  plt.figure()
  #plt.subplot(n_iter, 2, 2 * i + 2)
  plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i==0)
  if save_figures: pml.savefig('bayes-opt-acquisition-{}.pdf'.format(i+1))
  plt.show()
  
  # Add sample to previous samples
  X_sample = np.append(X_sample, np.atleast_2d(X_next), axis=0)
  Y_sample = np.append(Y_sample, np.atleast_2d(Y_next), axis=0)
    
  
def callback_noplot(X_next, Y_next, i):
  global X_sample, Y_sample
  X_next = np.atleast_2d(X_next)
  Y_next = np.atleast_2d(Y_next)
  X_sample = np.vstack((X_sample, X_next))
  Y_sample = np.vstack((Y_sample, Y_next))
  
n_restarts = 25
np.random.seed(0)
noise = 0.2
n_iter = 10
acq_fn = expected_improvement
acq_solver = MultiRestartGradientOptimizer(dim=1, bounds=bounds, n_restarts=n_restarts)
solver = BayesianOptimizer(X_init, Y_init, gpr, acq_fn, acq_solver, n_iter=n_iter, callback=callback)

solver.maximize(f)

 
plot_convergence(X_sample, Y_sample)
if save_figures: pml.savefig('bayes-opt-convergence.pdf')
plt.show()
  
####################
 # skopt, https://scikit-optimize.github.io/
"""
#from sklearn.base import clone
from skopt import gp_minimize

np.random.seed(0)
r = gp_minimize(lambda x: -f(np.array(x))[0], 
                bounds.tolist(),
                base_estimator=gpr,
                acq_func='EI',      # expected improvement
                xi=0.01,            # exploitation-exploration trade-off
                n_calls=10,         # number of iterations
                n_random_starts=0,  # initial samples are provided
                x0=X_init.tolist(), # initial samples
                y0=-Y_init.ravel())

# Fit GP model to samples for plotting results. Note negation of f.
gpr.fit(r.x_iters, -r.func_vals)
plot_approximation(gpr, X, Y, r.x_iters, -r.func_vals, show_legend=True)
save_fig('bayes-opt-skopt.pdf')
plt.show()

plot_convergence(np.array(r.x_iters), -r.func_vals)


###############
# https://github.com/SheffieldML/GPyOpt

import GPy
from GPyOpt.methods import BayesianOptimization

kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
bds = [{'name': 'X', 'type': 'continuous', 'domain': bounds.ravel()}]

np.random.seed(2)  
optimizer = BayesianOptimization(f=lambda X: -f(X),
                                 domain=bds,
                                 model_type='GP',
                                 kernel=kernel,
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.01,
                                 X=X_init,
                                 Y=-Y_init,
                                 noise_var = noise**2,
                                 exact_feval=False,
                                 normalize_Y=False,
                                 maximize=False)

optimizer.run_optimization(max_iter=10)
optimizer.plot_acquisition()
save_fig('bayes-opt-gpyopt.pdf')
plt.show()

optimizer.plot_convergence()


"""
