#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 04:46:34 2019

@author: kpmurphy
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt

def rnd_dna_string(N, alphabet='ACGT'):
    return ''.join([random.choice(alphabet) for i in range(N)])

def all_dna_strings(k):
    return (''.join(p) for p in itertools.product('ACGT', repeat=k))

def encode_dna(s):
  if s=='A':
    return 0
  if s=='C':
    return 1
  if s=='G':
    return 2
  if s=='T':
    return 3
  
def melting_temp_str(s):
  nA = sum(c=='A' for c in s)
  nC = sum(c=='C' for c in s)
  nG = sum(c=='G' for c in s)
  nT = sum(c=='T' for c in s)
  return 2*(nA+nT) + 4*(nG+nC)


def melting_temp_seq(x):
  nA = sum(c==0 for c in x)
  nC = sum(c==1 for c in x)
  nG = sum(c==2 for c in x)
  nT = sum(c==3 for c in x)
  return 2*(nA+nT) + 4*(nG+nC)

def temp_range(x):
  temp = melting_temp_seq(x)
  inrange = (temp*(12 <= temp)*(temp <= 16) + 
    temp*(20 <= temp)*(temp <= 22))
  return inrange


seq_len = 6 # L
embed_dim = 5 # D 
nhidden = 2
alpha_size = 4 # A
nseq = alpha_size ** seq_len
print("Generating {} sequences of length {}".format(nseq, seq_len))
S = list(all_dna_strings(seq_len)) # N-list of L-strings
S1 = [list(s) for s in S] # N-list of L-lists
S2 = np.array(S1) # N*L array of strings, N=A**L
X = np.vectorize(encode_dna)(S2) # N*L array of ints (in 0..A)

from sklearn.preprocessing import OneHotEncoder
cat = np.array(range(alpha_size)); 
cats = [cat]*np.shape(X)[1]
enc =  OneHotEncoder(sparse=False, categories=cats)
enc.fit(X)
Xhot = enc.transform(X)
Xcold = enc.inverse_transform(Xhot)
assert (Xcold==X).all()


def oracle_onehot_batch(Xhot):
  Xcold = enc.inverse_transform(Xhot)
  #return np.apply_along_axis(melting_temp_seq, 1,  Xcold)
  return np.apply_along_axis(temp_range, 1,  Xcold)
                          
y = oracle_onehot_batch(Xhot)

def build_model_hot():
  model = keras.Sequential([
      keras.layers.Dense(nhidden, activation=tf.nn.relu),
      keras.layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam(0.01)
  model.compile(optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_squared_error'])
  return model


"""
model_hot = build_model_hot()
model_hot.fit(Xhot, y, epochs=30)
pred_hot = model_hot.predict(Xhot)

plt.scatter(y, pred_hot)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('hot encoding')
plt.show()
"""

def build_model_cold():
  model = keras.Sequential([
      keras.layers.Embedding(alpha_size, embed_dim, input_length=seq_len),
      keras.layers.Flatten(input_shape=(seq_len, embed_dim)),
      keras.layers.Dense(nhidden, activation=tf.nn.relu),
      keras.layers.Dense(nhidden, activation=tf.nn.relu),
      keras.layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam(0.01)
  model.compile(optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_squared_error'])
  return model

  
model = build_model_cold()
model.fit(X, y, epochs=30)
pred = model.predict(X)


plt.scatter(y, pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('cold encoding')
plt.show()


def build_model_embed(model):
  embed = keras.Sequential([
      keras.layers.Embedding(alpha_size, embed_dim, input_length=seq_len,
                             weights=model.layers[0].get_weights()),
      keras.layers.Flatten(input_shape=(seq_len, embed_dim)),
      keras.layers.Dense(nhidden, activation=tf.nn.relu,
                         weights=model.layers[2].get_weights()),
      keras.layers.Dense(nhidden, activation=tf.nn.relu,
                         weights=model.layers[3].get_weights()),
      #keras.layers.Dense(1, weights=model.layers[3].get_weights())
  ])
  return embed

embedder = build_model_embed(model)
Z = embedder.predict(X)
N = np.shape(X)[0]
assert (np.shape(Z) == (N,nhidden))

plt.scatter(Z[:,0], Z[:,1], c=y)
plt.title('embeddings')
plt.colorbar()
plt.show()

