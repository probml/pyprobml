

import tensorflow as tf
from tensorflow import keras
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt

np.random.seed(0)

def gen_rnd_dna_string(N, alphabet='ACGT'):
    return ''.join([random.choice(alphabet) for i in range(N)])

def gen_all_dna_strings(L):
    return (''.join(p) for p in itertools.product('ACGT', repeat=L))

def encode_dna(s):
  if s=='A':
    return 0
  if s=='C':
    return 1
  if s=='G':
    return 2
  if s=='T':
    return 3
  
def gen_all_dna_seq(seq_len):
  S = list(gen_all_dna_strings(seq_len)) # N-list of L-strings
  S1 = [list(s) for s in S] # N-list of L-lists
  S2 = np.array(S1) # (N,L) array of strings, N=A**L
  X = np.vectorize(encode_dna)(S2) # (N,L) array of ints (in 0..A)
  return X


def motif_distance(x, m):
  # hamming distance of x to motif
  # If m[i]=nan, it means locn i is a don't care
  mask = [not(np.isnan(v)) for v in m] #np.where(m>0)
  return np.sum(x[mask] != m[mask])

def oracle(x):
  motifs = [ [np.nan, 0, 0, 1],
#           [1, 2, 3, np.nan]
            ];
  d = np.inf
  for motif in motifs:
    d = min(d, motif_distance(x, np.array(motif)))
  return d

#m=np.array([np.nan,1,2,3]); x=np.array([0,1,2,3]); motif_distance(x,m)

def oracle_batch(X):
  return np.apply_along_axis(oracle, 1,  X)

seq_len = 4 # L
embed_dim = 5 # D 
nhidden = 10
nlayers = 2
alpha_size = 4 # A
nseq = alpha_size ** seq_len
print("Generating {} sequences of length {}".format(nseq, seq_len))

X = gen_all_dna_seq(seq_len) # N*L array of ints (in 0..A)
y = oracle_batch(X)

xs = range(nseq);
nminima = np.size(np.where(y==0))
plt.figure(figsize=(10, 4))
plt.plot(xs, y);
plt.title("oracle has {} minima".format(nminima))
plt.show()

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
model.fit(X, y, epochs=50, verbose=1)
pred = model.predict(X)

plt.figure()
plt.scatter(y, pred)
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
Z = embedder.predict(X)
N = np.shape(X)[0]
assert (np.shape(Z) == (N,nhidden))

plt.figure()
plt.scatter(Z[:,0], Z[:,1], c=y)
plt.title('embeddings')
plt.colorbar()
plt.show()


from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
kernel_matrix = rbf_kernel(Z, gamma=1)
dist_matrix = pairwise_distances(Z)
#nearest = np.argsort(dist__matrix, axis=1)
Knn = 200 # 100
nearest = np.argpartition(dist_matrix, Knn+1, axis=1)

search = range(Knn);
for source in [0,1,2,3,4]:
  ysource = oracle(X[source])
  targets = nearest[source, search];
  #targets = np.argsort(abs(y-ysource))[:Knn] # cheating!
  dst = dist_matrix[source, targets];
  ytargets = oracle_batch(X[targets])
  """
  plt.figure()
  plt.plot(search, dst)
  plt.title('distance from {}'.format(source))
  
  plt.figure()
  plt.plot(search, ytargets-ysource)
  plt.title('relative value of nbrs')
  """
  plt.figure()
  plt.plot(dst, ytargets-ysource, 'o')
  plt.title('relative value of nbrs')
