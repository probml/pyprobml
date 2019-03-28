import numpy as np

from scipy.stats import norm


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


# Pass in all possible strings as Xall
# This defines the finite search space which we enumerate over.
# We embed each input using embed_fn before calling kernel.    
class BayesianOptimizerEmbedEnum(BayesianOptimizer):
  def __init__(self, Xall, embed_fn, 
               X_init, Y_init, surrogate, 
               acq_fn=expected_improvement, n_iter=None, callback=None,
               alphabet=[0,1,2,3]):
    self.embed_fn = embed_fn
    self.Xall = Xall
    self.Zcandidates = self.embed_fn(self.Xall)
    self.logging = []
    Z_init = self.embed_fn(X_init)
    super().__init__(Z_init, Y_init, surrogate, acq_fn=acq_fn,
         acq_solver=None, n_iter=n_iter, callback=callback)

  def propose(self):
    Zold = self.X_sample # already embedded
    A = self.acq_fn(self.Zcandidates, Zold, self.Y_sample, self.surrogate)
    ndxA = np.argmax(A)
    #### debugging
    current_iter = len(self.val_history)
    mu, sigma = self.surrogate.predict(self.Zcandidates, return_std=True)
    sigma = np.reshape(sigma, np.shape(mu))
    ndxY = np.argmax(mu)
    str = "Iter {}, Best acq {} val {:0.5f} surrogate {:0.5f} std {:0.3f}, best surrogate {} val {:0.5f}".format(
        current_iter, ndxA, A[ndxA], mu[ndxA], sigma[ndxA], ndxY, mu[ndxY])
    self.logging.append(str)
    #print(str)
    #plt.figure(figsize=(10,5)); plt.plot(A); plt.title('acq fn {}'.format(current_iter))
    #plt.figure(figsize=(10,5)); plt.plot(mu); plt.title('surrogate fn {}'.format(current_iter))
    #plt.figure(figsize=(10,5)); plt.plot(sigma); plt.title('sigma {}'.format(current_iter))
    ###
    return self.Xall[ndxA]
  
  def update(self, x, y):
    X = np.atleast_2d(x)
    Z = self.embed_fn(X)
    self.X_sample = np.append(self.X_sample, Z, axis=0) # store embeddings
    self.Y_sample = np.append(self.Y_sample, y)
    self.surrogate.fit(self.X_sample, self.Y_sample)
    if y > self.current_best_val:
      self.current_best_arg = x
      self.current_best_val = y
    self.val_history = np.append(self.val_history, y)
  
# Soecify the set of possible strings as Xall
class DiscreteOptimizer:
  def __init__(self, Xall,
               n_iter=None, callback=None):
    self.Xall = Xall
    self.current_best_arg = None
    self.current_best_val = -np.inf
    self.n_iter = n_iter
    self.callback = callback
    self.val_history = []
    
  def propose(self):
    pass
  
  def update(self, x, y):
    if y > self.current_best_val:
      self.current_best_arg = x
      self.current_best_val = y
    self.val_history = np.append(self.val_history, y)
      
  def maximize(self, objective):
    for i in range(self.n_iter):
      X_next = self.propose()
      Y_next = objective(X_next)
      self.update(X_next, Y_next)
      if self.callback is not None:
        self.callback(X_next, Y_next, i)
    return self.current_best_arg
  

class EnumerativeDiscreteOptimizer(DiscreteOptimizer):
  def __init__(self, Xall,
               n_iter=None, callback=None):
    super().__init__(Xall, n_iter, callback)
    self.ndx = 0
    
  def propose(self):
    x = self.Xall[self.ndx]
    n = np.shape(self.Xall)[0]
    if self.ndx == n-1:
      self.ndx = 0
    else:
      self.ndx += 1
    return x
  
  def maximize(self, objective):
    self.ndx = 0
    return super().maximize(objective)
  
  
class RandomDiscreteOptimizer(DiscreteOptimizer):
  def __init__(self, Xall,
               n_iter=None, callback=None):
    super().__init__(Xall, n_iter, callback)
    
  def propose(self):
    #x = gen_rnd_string(self.seq_len, self.alphabet)
    n = np.shape(self.Xall)[0]
    ndx = np.random.randint(low=0, high=n, size=1)[0]
    x = self.Xall[ndx]
    return x
  

  
  


