import numpy as np
from scipy.stats import norm
  
#from scipy.optimize import minimize
import scipy.optimize

def expected_improvement(X, X_sample, Y_sample, surrogate, xi=0.01, noise_free=False):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a probabilistic surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        surrogate: a model with a predict that returns mu, sigma
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = surrogate.predict(X, return_std=True)
    sigma = sigma.reshape(-1, X_sample.shape[1])
    
    if noise_free:
      current_best = np.max(Y_sample)
    else:
      mu_sample = surrogate.predict(X_sample)
      current_best = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - current_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


# bounds: D*2 array, where D = number parameter dimensions
# bounds[:,0] are lower bounds, bounds[:,1] are upper bounds
class GradientOptimizer:
  def __init__(self, dim, bounds=None, n_restarts=1, method='L-BFGS-B'):
    self.bounds = bounds
    self.n_restarts = n_restarts
    self.method = method
    self.dim = dim
  
  
  def maximize(self, objective):
    neg_obj = lambda x: -objective(x)
    min_val = np.inf
    best_x = None
    candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                   size=(self.n_restarts, self.dim))
    for x0 in candidates:
        res = scipy.optimize.minimize(neg_obj, x0=x0, bounds=self.bounds,
                                     method=self.method)        
        if res.fun < min_val:
            min_val = res.fun[0]
            best_x = res.x 
    return best_x.reshape(-1, 1)
 
class RandomOptimizer:
  def __init__(self, dim, bounds=None, n_restarts=10):
    self.bounds = bounds
    self.n_restarts = n_restarts
    self.dim = dim
  
  def maximize(self, objective):
    neg_obj = lambda x: -objective(x)
    min_val = np.inf
    best_x = None
    # Find the best optimum by starting from n_restart different random points.
    candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                   size=(self.n_restarts, self.dim))
    for x0 in candidates:
      res = neg_obj(x0)
      if res < min_val:
        min_val = res
        best_x = x0
    return best_x.reshape(-1, 1)
  
class BayesianOptimizer:
  def __init__(self, X_init, Y_init, surrogate, 
               acq_fn=expected_improvement, acq_solver=None):
    self.current_best_arg = None
    self.current_best_val = np.inf
    self.X_sample = X_init
    self.Y_sample = Y_init
    self.surrogate = surrogate
    self.surrogate.fit(self.X_sample, self.Y_sample)
    self.acq_fn = acq_fn
    self.acq_solver = acq_solver
  
  def propose(self):
    def objective(X):
      dim = self.X_sample.shape[1]
      X = X.reshape(-1, dim) # make vector into matrix
      return self.acq_fn(X, self.X_sample, self.Y_sample, self.surrogate)
    X_next = self.acq_solver.maximize(objective)
    return X_next
 
  def update(self, x, y):
    self.X_sample = np.append(self.X_sample, x, axis=0)
    self.Y_sample = np.append(self.Y_sample, y, axis=0)
    self.surrogate.fit(self.X_sample, self.Y_sample)
    if y > self.current_best_val:
      self.current_best_arg = x
      self.current_best_val = y
      
  def current_best(self):
    return (self.current_best_arg, self.current_best_val)

##########    

from sklearn.gaussian_process.kernels import Matern
# https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/gaussian_process/kernels.py#L1146
class EmbedKernel(Matern):
  def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 nu=1.5, embed_fn=None):
        super().__init__(length_scale, length_scale_bounds)
        self.embed_fn = embed_fn
 
  def __call__(self, X, Y=None, eval_gradient=False):
    if self.embed_fn is not None:
      X = self.embed_fn(X)
      if Y is not None:
        Y = self.embed_fn(Y)
    return super().__call__(X, Y=Y, eval_gradient=eval_gradient)

