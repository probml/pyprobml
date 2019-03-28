#import numpy as np

import tensorflow as tf
from tensorflow import keras

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
