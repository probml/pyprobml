import numpy as np

import tensorflow as tf
from tensorflow import keras
 

def build_supervised_model(seq_len, nlayers):
  embed_dim = 5 # D 
  nhidden = 10
  alpha_size = 4
  model = keras.Sequential()
  model.add(keras.layers.Embedding(alpha_size, embed_dim, input_length=seq_len))
  model.add(keras.layers.Flatten(input_shape=(seq_len, embed_dim)))
  for l in range(nlayers):
      model.add(keras.layers.Dense(
          nhidden, activation=tf.nn.relu,
          kernel_regularizer=keras.regularizers.l2(0.0001)))
  model.add(keras.layers.Dense(1))
  optimizer = tf.keras.optimizers.Adam(0.01)
  model.compile(optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_squared_error'])
  return model
  
def learn_supervised_model(Xtrain, ytrain, nlayers, epochs=20):
  seq_len = np.shape(Xtrain)[1]
  model = build_supervised_model(seq_len, nlayers)
  model.fit(Xtrain, ytrain, epochs=epochs, verbose=1, batch_size=32)
  return model 

def convert_supervised_to_embedder(model, seq_len, nlayers):
  embed_dim = 5 # D 
  nhidden = 10
  alpha_size = 4
  embed = keras.Sequential()
  embed.add(keras.layers.Embedding(alpha_size, embed_dim, input_length=seq_len,
                             weights=model.layers[0].get_weights()))
  embed.add(keras.layers.Flatten(input_shape=(seq_len, embed_dim)),)
  for l in range(nlayers):
      embed.add(keras.layers.Dense(nhidden, activation=tf.nn.relu,
                         weights=model.layers[2+l].get_weights()))

  return embed

