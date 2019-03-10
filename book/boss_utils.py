import numpy as np

from bayes_opt_utils import BayesianOptimizer, expected_improvement
from bayes_opt_utils import EnumerativeStringOptimizer, RandomStringOptimizer

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from bayes_opt_utils import EmbedKernel

import tensorflow as tf
from tensorflow import keras
 
def callback_logger(xnext, ynext, i):
  global ytrace
  print("iter {}, x={}, y={}".format(i, xnext, ynext))
  current_best = np.max(ytrace)
  if ynext > current_best:
    ytrace = np.append(ytrace, ynext)
  else:
    ytrace = np.append(ytrace, current_best)  

def boss_maximize_bayes(oracle, Xinit, yinit, embed_fn, n_iter):
  global ytrace
  ytrace = [np.max(yinit)]
  seq_len = np.shape(Xinit)[1]
  kernel = ConstantKernel(1.0) * EmbedKernel(length_scale=1.0, nu=2.5, embed_fn=embed_fn)
  noise = np.std(yinit)
  gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
  acq_fn = expected_improvement
  n_seq = 4**seq_len
  acq_solver =  EnumerativeStringOptimizer(seq_len, n_iter=n_seq)
  solver = BayesianOptimizer(Xinit, yinit, gpr, acq_fn, acq_solver, n_iter=n_iter, callback=callback_logger)
  solver.maximize(oracle)
  return ytrace

def boss_maximize_random(oracle, Xinit, yinit, embed_fn, n_iter):
  global ytrace
  ytrace = [np.max(yinit)]
  seq_len = np.shape(Xinit)[1]
  solver = RandomStringOptimizer(seq_len, n_iter=n_iter, callback=callback_logger)
  solver.maximize(oracle)
  return ytrace

def boss_maximize(method, oracle, Xinit, yinit, embed_fn, n_iter):
  if method=='random':
    return boss_maximize_random(oracle, Xinit, yinit, embed_fn, n_iter)
  if method=='bayes':
    return boss_maximize_bayes(oracle, Xinit, yinit, embed_fn, n_iter)






def build_supervised_model(seq_len):
  embed_dim = 5 # D 
  nhidden = 10
  nlayers = 2
  alpha_size = 4
  model = keras.Sequential()
  model.add(keras.layers.Embedding(alpha_size, embed_dim, input_length=seq_len))
  model.add(keras.layers.Flatten(input_shape=(seq_len, embed_dim)))
  for l in range(nlayers):
      model.add(keras.layers.Dense(nhidden, activation=tf.nn.relu))
  model.add(keras.layers.Dense(1))
  optimizer = tf.keras.optimizers.Adam(0.01)
  model.compile(optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_squared_error'])
  return model
  
def learn_supervised_model(Xtrain, ytrain):
  seq_len = np.shape(Xtrain)[1]
  model = build_supervised_model(seq_len)
  model.fit(Xtrain, ytrain, epochs=20, verbose=1, batch_size=32)
  return model 

def convert_to_embedder(model, seq_len):
  ninclude = 2
  embed_dim = 5 # D 
  nhidden = 10
  alpha_size = 4
  embed = keras.Sequential()
  embed.add(keras.layers.Embedding(alpha_size, embed_dim, input_length=seq_len,
                             weights=model.layers[0].get_weights()))
  embed.add(keras.layers.Flatten(input_shape=(seq_len, embed_dim)),)
  for l in range(ninclude):
      embed.add(keras.layers.Dense(nhidden, activation=tf.nn.relu,
                         weights=model.layers[2+l].get_weights()))

  return embed

