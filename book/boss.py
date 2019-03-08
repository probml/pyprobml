

import tensorflow as tf
from tensorflow import keras
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(0)


def gen_rnd_dna(seq_len):
  s = [random.choice([0,1,2,3]) for i in range(seq_len)]
  return np.array(s)

def gen_all_dna(seq_len):
  S = [np.array(p) for p in itertools.product([0,1,2,3], repeat=seq_len)]
  return np.stack(S)

def motif_distance(x, m):
  # hamming distance of x to motif
  # If m[i]=nan, it means locn i is a don't care
  mask = [not(np.isnan(v)) for v in m] #np.where(m>0)
  return np.sum(x[mask] != m[mask])



seq_len = 6 # L
alpha_size = 4 # A
nseq = alpha_size ** seq_len
print("Generating {} sequences of length {}".format(nseq, seq_len))

motifs = [];
#m = np.arange(seq_len, dtype=float)
m = np.repeat(0.0, seq_len)
m1 = np.copy(m)
m1[0] = np.nan
m2 = np.copy(m)
m2[seq_len-1] = np.nan
#motifs = [m1, m2]
motifs = [m2]
print("Motifs")
print(motifs)
  
def oracle(x):
  d = np.inf
  for motif in motifs:
    d = min(d, motif_distance(x, motif))
  return d

#m=np.array([np.nan,1,2,3]); x=np.array([0,1,2,3]); motif_distance(x,m)

def oracle_batch(X):
  return np.apply_along_axis(oracle, 1,  X)

Xall = gen_all_dna(seq_len) # (N,L) array of ints (in 0..A)
yall = oracle_batch(Xall)

#np.where((X==[0,0,0,1,1,0]).all(axis=1))

nminima = np.size(np.where(yall==min(yall)))
plt.figure(figsize=(10, 4))
plt.plot(range(nseq), yall);
plt.title("oracle has {} minima".format(nminima))
plt.show()

# Extract training set based on "medium performing" strings
bins = pd.qcut(yall, 4, labels=False, duplicates='drop')
middle_bins = np.where(np.logical_or(bins==1, bins==2))
Xtrain = Xall[middle_bins]
ytrain = yall[middle_bins]
ntrain = np.shape(Xtrain)[0]
print("Training set has {} examples from {}".format(ntrain, nseq))

embed_dim = 5 # D 
nhidden = 10
nlayers = 2
def build_model():
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
  
model = build_model()
model.fit(Xtrain, ytrain, epochs=10, verbose=1)
ypred = model.predict(Xall)

plt.figure()
plt.scatter(yall, ypred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()


def build_model_embed(model, ninclude_layers=nlayers):
  embed = keras.Sequential()
  embed.add(keras.layers.Embedding(alpha_size, embed_dim, input_length=seq_len,
                             weights=model.layers[0].get_weights()))
  embed.add(keras.layers.Flatten(input_shape=(seq_len, embed_dim)),)
  for l in range(ninclude_layers):
      embed.add(keras.layers.Dense(nhidden, activation=tf.nn.relu,
                         weights=model.layers[2+l].get_weights()))

  return embed

embedder = build_model_embed(model, 1)
Z = embedder.predict(Xtrain)
plt.figure()
plt.scatter(Z[:,1], Z[:,2], c=ytrain)
plt.title('embeddings of training set')
plt.colorbar()
plt.show()


#from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import pairwise_distances
#kernel_matrix = rbf_kernel(Z, gamma=1)
#dist_matrix = pairwise_distances(Z)
#nearest = np.argsort(dist__matrix, axis=1)

sources = np.arange(4)
dist_matrix = pairwise_distances(Z[sources], Z)
nearest = np.argsort(dist_matrix, axis=1)
knn = 100
#fig, ax = plt.subplots(2,2)
for source, ndx in enumerate(sources):
  ysource = oracle(Xall[source])
  nbrs = nearest[source, 0:knn];
  #targets = np.argsort(abs(y-ysource))[:Knn] # cheating!
  dst = dist_matrix[source, nbrs];
  ytargets = oracle_batch(Xall[nbrs])
  plt.figure()
  plt.plot(dst, ytargets-ysource, 'o')
  plt.title('f-value of nbrs relative to source {}'.format(source))
plt.show()

    
class EnumerativeSolver:
  def __init__(self, seq_len):
    self.seq_len = seq_len
    Xall = gen_all_dna(seq_len) # could use iterator
    nseq = np.shape(Xall)[0]
    perm = np.random.permutation(nseq)
    self.Xall = Xall[perm]
    self.ndx = 0
    self.current_best_seq = None
    self.current_best_val = np.inf
  
  def propose(self):
    x = self.Xall[self.ndx]
    if self.ndx == nseq:
      self.ndx = 0
    else:
      self.ndx += 1
    return x
  
  def update(self, x, y):
    if y < self.current_best_val:
      self.current_best_seq = x
      self.current_best_val = y
      
  def current_best(self):
    return (self.current_best_seq, self.current_best_val)


class GPSolver:
  def __init__(self, seq_len, ninit):
    self.seq_len = seq_len
    self.ninit = ninit
    self.current_best_seq = None
    self.current_best_val = np.inf
    self.nqueries = 0
  
  def propose(self):
    return gen_rnd_dna(self.seq_len)
 
  
  def update(self, x, y):
    if y < self.current_best_val:
      self.current_best_seq = x
      self.current_best_val = y
      
  def current_best(self):
    return (self.current_best_seq, self.current_best_val)
  
nsteps = 100
solver = EnumerativeSolver(seq_len)
history = []
for t in range(nsteps):
  x = solver.propose()
  y = oracle(x)
  solver.update(x, y)
  xbest, ybest = solver.current_best()
  history.append(ybest)
  
plt.figure()
plt.plot(range(nsteps), history)
            
      
      