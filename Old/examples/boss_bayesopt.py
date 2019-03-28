import numpy as np

import tensorflow as tf
from tensorflow import keras
 
from utils import gen_rnd_string, gen_all_strings
from bayes_opt_utils import BayesianOptimizer, expected_improvement

def build_supervised_model(hp):
  alpha_size = 4
  model = keras.Sequential()
  model.add(keras.layers.Embedding(alpha_size, hp['embed_dim'], input_length=hp['seq_len']))
  model.add(keras.layers.Flatten(input_shape=(hp['seq_len'], hp['embed_dim'])))
  for l in range(hp['nlayers']):
      model.add(keras.layers.Dense(
          hp['nhidden'], activation=tf.nn.relu,
          kernel_regularizer=keras.regularizers.l2(0.0001)))
  model.add(keras.layers.Dense(1))
  optimizer = tf.keras.optimizers.Adam(0.01)
  model.compile(optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_squared_error'])
  return model
  
def learn_supervised_model(Xtrain, ytrain, hparams):
  model = build_supervised_model(hparams)
  model.fit(Xtrain, ytrain, epochs=hparams['epochs'], verbose=1, batch_size=32)
  return model 

def convert_supervised_to_embedder(model, hp, nlayers=None):
  if nlayers is None:
    nlayers = hp['nlayers']
  alpha_size = 4
  embed = keras.Sequential()
  embed.add(keras.layers.Embedding(alpha_size, hp['embed_dim'], input_length=hp['seq_len'],
                             weights=model.layers[0].get_weights()))
  embed.add(keras.layers.Flatten(input_shape=(hp['seq_len'], hp['embed_dim'])),)
  for l in range(nlayers):
      embed.add(keras.layers.Dense(hp['nhidden'], activation=tf.nn.relu,
                         weights=model.layers[2+l].get_weights()))

  return embed

import scipy.optimtize
class MultiRestartGradientOptimizerScalar:
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
          min_val = res.fun
          best_x = res.x 
    return best_x


class BayesianOptimizerEmbedEnum(BayesianOptimizer):
  def __init__(self, Xall, embed_fn, 
               X_init, Y_init, surrogate, 
               acq_fn=expected_improvement, n_iter=None, callback=None,
               alphabet=[0,1,2,3]):
    self.embed_fn = embed_fn
    self.Xall = Xall
    self.logging = []
    Z_init = self.embed_fn(X_init)
    super().__init__(Z_init, Y_init, surrogate, acq_fn=acq_fn,
         acq_solver=None, n_iter=n_iter, callback=callback)

  def propose(self):
    Zcandidates = self.embed_fn(self.Xall)
    Zold = self.X_sample # already embedded
    A = self.acq_fn(Zcandidates, Zold, self.Y_sample, self.surrogate)
    ndxA = np.argmax(A)
    #### debugging
    current_iter = len(self.val_history)
    mu, sigma = self.surrogate.predict(Zcandidates, return_std=True)
    sigma = np.reshape(sigma, np.shape(mu))
    ndxY = np.argmax(mu)
    str = "Iter {}, Best acq {} val {:0.5f} surrogate {:0.5f} std {:0.3f}, best surrogate {} val {:0.5f}".format(
        current_iter, ndxA, A[ndxA], mu[ndxA], sigma[ndxA], ndxY, mu[ndxY])
    self.logging.append(str)
    #plt.figure(figsize=(10,5)); plt.plot(A); plt.title('acq fn {}'.format(current_iter))
    #plt.figure(figsize=(10,5)); plt.plot(mu); plt.title('surrogate fn {}'.format(current_iter))
    #plt.figure(figsize=(10,5)); plt.plot(sigma); plt.title('sigma {}'.format(current_iter))
    ###
    return self.Xall[ndxA]
  
  def update(self, x, y):
    X = np.atleast_2d(x)
    Z = self.embed_fn(X)
    self.X_sample = np.append(self.X_sample, Z, axis=0)
    self.Y_sample = np.append(self.Y_sample, y)
    self.surrogate.fit(self.X_sample, self.Y_sample)
    if y > self.current_best_val:
      self.current_best_arg = x
      self.current_best_val = y
    self.val_history = np.append(self.val_history, y)
  
  
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
    ndx = np.random.randint(low=0, high=n, size=1)
    x = self.Xall[ndx]
    return x
  

  
  
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

