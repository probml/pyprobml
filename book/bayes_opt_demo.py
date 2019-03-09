# Bayesian optimization of 1d continuous function
# Modified from Martin Krasser's code
# https://github.com/krasserm/bayesian-machine-learning/blob/master/bayesian_optimization.ipynb
# Apache 2.0 license

import numpy as np
import matplotlib.pyplot as plt
from utils import save_fig

np.random.seed(0)


def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
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
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
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
plt.legend();
save_fig('bayes-opt-init.pdf')
plt.show()

from scipy.stats import norm

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01, noise_free=False):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    sigma = sigma.reshape(-1, X_sample.shape[1])
    
    if noise_free:
      current_best = np.max(Y_sample)
    else:
      mu_sample = gpr.predict(X_sample)
      current_best = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - current_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei
  
from scipy.optimize import minimize

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = np.inf
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)


class GPMaximizer:
  def __init__(self, X_init, Y_init, gpr, bounds, n_restarts):
    self.current_best_arg = None
    self.current_best_val = np.inf
    self.X_sample = X_init
    self.Y_sample = Y_init
    self.gpr = gpr
    self.gpr.fit(self.X_sample, self.Y_sample)
    self.n_restarts = n_restarts
    self.bounds = bounds
  
  def propose(self):
    X_next = propose_location(
        expected_improvement, self.X_sample, self.Y_sample, self.gpr,
        self.bounds, self.n_restarts)
    return X_next
 
  def update(self, x, y):
    self.X_sample = np.append(self.X_sample, x, axis=0)
    self.Y_sample = np.append(self.Y_sample, y, axis=0)
    self.gpr.fit(self.X_sample, self.Y_sample)
    if y > self.current_best_val:
      self.current_best_arg = x
      self.current_best_val = y
      
  def current_best(self):
    return (self.current_best_arg, self.current_best_val)

##########    
    
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/gaussian_process/kernels.py#L1146
class StringEmbedKernel(Matern):
  def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 nu=1.5, seq_len=None):
        super().__init__(length_scale, length_scale_bounds)
        self.seq_len = seq_len
 
  def __call__(self, X, Y=None, eval_gradient=False):
    print("seq len = {}".format(self.seq_len))
    return super().__call__(X, Y=Y, eval_gradient=eval_gradient)


################
    
kernel = ConstantKernel(1.0) * StringEmbedKernel(length_scale=1.0, nu=2.5, seq_len=42)
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)


n_restarts = 25
np.random.seed(0)
noise = 0.2
solver = GPMaximizer(X_init, Y_init, gpr, bounds, n_restarts)
n_iter = 10
for i in range(n_iter):
    # Extract current data for plotting  
    X_sample = solver.X_sample
    Y_sample = solver.Y_sample
    
    X_next = solver.propose()
    Y_next = f(X_next, noise)
    solver.update(X_next, Y_next)
    
    # Plot samples, surrogate function, noise-free objective and next sampling location
    #plt.subplot(n_iter, 2, 2 * i + 1)
    plt.figure()
    plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i==0)
    plt.title(f'Iteration {i+1}')
    save_fig('bayes-opt-surrogate-{}.pdf'.format(i+1))
    plt.show()
    
    plt.figure()
    #plt.subplot(n_iter, 2, 2 * i + 2)
    plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i==0)
    save_fig('bayes-opt-acquisition-{}.pdf'.format(i+1))
    plt.show()
  
plot_convergence(solver.X_sample, solver.Y_sample)
save_fig('bayes-opt-convergence.pdf')
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
