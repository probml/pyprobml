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


seq_len = 3 # L
embed_dim = 4# D 
alpha_size = 4 # A

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
  return np.apply_along_axis(melting_temp_seq, 1,  Xcold)
                          
y = oracle_onehot_batch(Xhot)

def build_model():
  model = keras.Sequential([
      keras.layers.Embedding(alpha_size, embed_dim, input_length=seq_len),
      keras.layers.Flatten(input_shape=(seq_len, embed_dim)),
      keras.layers.Dense(2, activation=tf.nn.relu),
      keras.layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.Adam(0.01)
  model.compile(optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_squared_error'])
  return model

  
model = build_model()
model.fit(X, y, epochs=50)
pred = model.predict(X)

plt.scatter(y, pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')


"""
model = keras.Sequential()
model.add(keras.layers.Embedding(alpha_size, embed_dim, input_length=seq_len))
#N = 32
#input_array = np.random.randint(alpha_size, size=(N, seq_len))
input_array = X
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
N = np.shape(input_array)[0]
assert output_array.shape == (N, seq_len, embed_dim)
"""


