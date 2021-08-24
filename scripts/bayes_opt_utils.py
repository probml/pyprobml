import superimport

import numpy as np
from scipy.stats import norm

#from scipy.optimize import minimize
import scipy.optimize

def expected_improvement(X, X_sample, Y_sample, surrogate,
                         improvement_thresh=0.01, trust_incumbent=False,
                         greedy=False):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a probabilistic surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        surrogate: a model with a predict that returns mu, sigma
        improvement_thresh: Exploitation-exploration trade-off parameter.
        trust_incumbent: whether to trust current best obs. or re-evaluate
    
    Returns:
        Expected improvements at points X.
    '''
    #X = np.atleast_2d(X)
    mu, sigma = surrogate.predict(X, return_std=True)
    # Make sigma have same shape as mu
    #sigma = sigma.reshape(-1, X_sample.shape[1])
    sigma = np.reshape(sigma, np.shape(mu))
    
    if trust_incumbent:
      current_best = np.max(Y_sample)
    else:
      mu_sample = surrogate.predict(X_sample)
      current_best = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - current_best - improvement_thresh
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        #ei[sigma == 0.0] = 0.0
        ei[sigma < 1e-4] = 0.0
      
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
    #neg_obj = lambda x: -objective(x)
    neg_obj = lambda x: -objective(x.reshape(-1, 1))
    min_val = np.inf
    best_x = None
    candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                   size=(self.n_restarts, self.dim))
    for x0 in candidates:
        res = scipy.optimize.minimize(neg_obj, x0=x0, bounds=self.bounds,
                                     method=self.method)        
        if res.fun < min_val:
          min_val = res.fun
          best_x = res.x 
    return best_x.reshape(-1, 1)

 
  
class BayesianOptimizer:
  def __init__(self, X_init, Y_init, surrogate, 
               acq_fn=expected_improvement, acq_solver=None,
               n_iter=None, callback=None):
    self.X_sample = X_init
    self.Y_sample = Y_init
    self.surrogate = surrogate
    self.surrogate.fit(self.X_sample, self.Y_sample)
    self.acq_fn = acq_fn
    self.acq_solver = acq_solver
    self.n_iter = n_iter
    self.callback = callback
    # Make sure you "pay" for the initial random guesses
    self.val_history = Y_init
    self.current_best_val = np.max(self.val_history)
    best_ndx = np.argmax(self.val_history)
    self.current_best_arg = X_init[best_ndx]
  
  def propose(self):
    def objective(x):
      y = self.acq_fn(x, self.X_sample, self.Y_sample, self.surrogate)
      if np.size(y)==1:
        y = y[0] # convert to scalar
      return y
    x_next = self.acq_solver.maximize(objective)
    return x_next
 
  def update(self, x, y):
    X = np.atleast_2d(x)
    self.X_sample = np.append(self.X_sample, X, axis=0)
    self.Y_sample = np.append(self.Y_sample, y)
    self.surrogate.fit(self.X_sample, self.Y_sample)
    if y > self.current_best_val:
      self.current_best_arg = x
      self.current_best_val = y
    self.val_history = np.append(self.val_history, y)
    
  def maximize(self, objective):
    for i in range(self.n_iter):
      X_next = self.propose()
      Y_next = objective(X_next)
      #print("BO iter {}, xnext={}, ynext={:0.3f}".format(i, X_next, Y_next))
      self.update(X_next, Y_next)
      if self.callback is not None:
        self.callback(X_next, Y_next, i)
    return self.current_best_arg

