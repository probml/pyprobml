import numpy as np
from scipy.stats import norm
from utils import gen_rnd_string, gen_all_strings

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
    X = np.atleast_2d(X)
    mu, sigma = surrogate.predict(X, return_std=True)
    # Make sigma have same shape as mu
    #sigma = sigma.reshape(-1, X_sample.shape[1])
    sigma = np.reshape(sigma, np.shape(mu))
    
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
class MultiRestartGradientOptimizer:
  def __init__(self, dim, bounds=None, n_restarts=1, method='L-BFGS-B',
               callback=None):
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
    return best_x

 
class RandomOptimizer:
  def __init__(self, dim, bounds=None, n_samples=None, callback=None):
    self.bounds = bounds
    self.n_samples = n_samples
    self.dim = dim
    self.callback = callback
  
  def maximize(self, objective):
    min_val = np.inf
    best_x = None
    # Find the best optimum by starting from n_restart different random points.
    candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                   size=(self.n_samples, self.dim))
    for i, x in enumerate(candidates):
      y = objective(x)
      if self.callback is not None:
        self.callback(x, y, i)
      if y < min_val:
        min_val = y
        best_x = x
    return best_x

   

class BayesianOptimizer:
  def __init__(self, X_init, Y_init, surrogate, 
               acq_fn=expected_improvement, acq_solver=None,
               n_iter=None, callback=None):
    self.current_best_arg = None
    self.current_best_val = np.inf
    self.X_sample = X_init
    self.Y_sample = Y_init
    self.surrogate = surrogate
    self.surrogate.fit(self.X_sample, self.Y_sample)
    self.acq_fn = acq_fn
    self.acq_solver = acq_solver
    self.n_iter = n_iter
    self.callback = callback
  
  def propose(self):
    def objective(x):
      return self.acq_fn(x, self.X_sample, self.Y_sample, self.surrogate)
    x_next = self.acq_solver.maximize(objective)
    return x_next
 
  def update(self, x, y):
    X = np.atleast_2d(x)
    Y = np.atleast_2d(y)
    self.X_sample = np.append(self.X_sample, X, axis=0)
    self.Y_sample = np.append(self.Y_sample, Y, axis=0)
    self.surrogate.fit(self.X_sample, self.Y_sample)
    if y > self.current_best_val:
      self.current_best_arg = x
      self.current_best_val = y
      
  def maximize(self, objective):
    for i in range(self.n_iter):
      X_next = self.propose()
      Y_next = objective(X_next)
      self.update(X_next, Y_next)
      if self.callback is not None:
        self.callback(X_next, Y_next, i)
    return self.current_best_arg, self.current_best_val


class EnumerativeStringOptimizer:
  def __init__(self, seq_len, alphabet=[0,1,2,3]):
    self.seq_len = seq_len
    self.Xall = gen_all_strings(seq_len, alphabet) # could use iterator
    self.ndx = 0
    self.current_best_arg = None
    self.current_best_val = np.inf
  
  def propose(self):
    x = self.Xall[self.ndx]
    nseq = np.size(self.Xall)[0]
    if self.ndx == nseq:
      self.ndx = 0
    else:
      self.ndx += 1
    return x
  
  def update(self, x, y):
    if y > self.current_best_val:
      self.current_best_arg = x
      self.current_best_val = y
      
  
  
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

