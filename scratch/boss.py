

import tensorflow as tf
from tensorflow import keras
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(0)

def encode_dna(s):
  if s=='A':
    return 0
  if s=='C':
    return 1
  if s=='G':
    return 2
  if s=='T':
    return 3
  
def encode_data(S):
  # S is an N-list of L-strings, L=8, N=65536
  S1 = [list(s) for s in S] # N-list of L-lists
  S2 = np.array(S1) # (N,L) array of strings, N=A**L
  X = np.vectorize(encode_dna)(S2) # (N,L) array of ints (in 0..A)
  return X

def decode_dna(x):
  alpha = ['A', 'C', 'G', 'T']
  return alpha[x]

def decode_data(X):
  S = np.vectorize(decode_dna)(X)
  return S

def zscore_normalize(data):
  return (data - data.mean()) / np.maximum(data.std(), 1e-8)

def min_max_normalize(data):
  return (data - data.min()) / np.maximum(data.max() - data.min(), 1e-8)

def get_8mer_data():
  file_name = '/home/kpmurphy/github/pyprobml/data/8mers_crx_ref_r1.csv'
  data = pd.read_csv(file_name, sep='\t')
  S = data['seq'].values
  y = data['val'].values
  X = encode_data(S)
  y = zscore_normalize(y)
  return X, y

Xall, yall = get_8mer_data()
nseq, seq_len = np.shape(Xall)
alpha_size = 4

def oracle(x):
  ndx = np.where((Xall==x).all(axis=1))
  return yall[ndx]

def oracle_batch(X):
  return np.apply_along_axis(oracle, 1,  X)

def gen_rnd_dna(seq_len):
  s = [random.choice([0,1,2,3]) for i in range(seq_len)]
  return np.array(s)

def gen_all_dna(seq_len):
  S = [np.array(p) for p in itertools.product([0,1,2,3], repeat=seq_len)]
  return np.stack(S)


plt.figure()
plt.plot(yall)

# Extract training set based on "medium performing" strings
# These could be unlabeled (if we use an RNN feature extractor)
bins = pd.qcut(yall, 100, labels=False, duplicates='drop')
middle_bins = np.where(np.logical_and(bins>=25, bins<=75))
Xtrain = Xall[middle_bins]
ytrain = yall[middle_bins]
ntrain = np.shape(Xtrain)[0]
print("Training set has {} examples from {}".format(ntrain, nseq))

# Pick a small labeled subset for training the GP
perm = np.random.permutation(ntrain)
ninit = 10
perm = perm[:ninit]
Xinit = Xtrain[perm]
yinit = ytrain[perm]

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
model.fit(Xtrain, ytrain, epochs=30, verbose=1, batch_size=32)
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
plt.scatter(Z[:,0], Z[:,1], c=ytrain)
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
fig, ax = plt.subplots(2,2)
for i, source in enumerate(sources):
  ysource = oracle(Xall[source])
  nbrs = nearest[source, 0:knn];
  dst = dist_matrix[source, nbrs];
  ytargets = oracle_batch(Xall[nbrs])
  #plt.figure()
  r = i // 2
  c = i % 2
  ax[r,c].plot(dst, ytargets-ysource, 'o')
  ax[r,c].set_title('source {}'.format(source))
plt.show()

    
class EnumerativeSolver:
  def __init__(self, seq_len):
    self.seq_len = seq_len
    Xall = gen_all_dna(seq_len) # could use iterator
    nseq = np.shape(Xall)[0]
    # We enumerate them in a random order, in case the optimal sequence
    # is near the start.
    perm = np.random.permutation(nseq)
    self.Xall = Xall[perm]
    self.ndx = 0
    self.current_best_seq = None
    self.current_best_val = np.inf
    self.nqueries = 0
  
  def propose(self):
    self.nqueries += 1
    x = self.Xall[self.ndx]
    if self.ndx == nseq:
      self.ndx = 0
    else:
      self.ndx += 1
    return x
  
  def update(self, x, y):
    if y > self.current_best_val:
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
plt.title('Enumerative solver')





from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern, RBF

class StringEmbedKernel(Matern):
  def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 nu=1.5, seq_len=None, embedder=None):
        super().__init__(length_scale, length_scale_bounds)
        self.seq_len = seq_len
        self.embedder = embedder
 
  def __call__(self, X, Y=None, eval_gradient=False):
    print("seq len = {}".format(self.seq_len))
    X = self.embedder(X)
    if Y is not None:
      Y = self.embedder(Y)
    return super().__call__(X, Y=Y, eval_gradient=eval_gradient)

"""
https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/gaussian_process/kernels.py#L1287
The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.
"""

# https://github.com/krasserm/bayesian-machine-learning/blob/master/bayesian_optimization.ipynb
from scipy.stats import norm
def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    #sigma = sigma.reshape(-1, X_sample.shape[1])
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei
  
class GPMaximizer:
  def __init__(self, seq_len, embedder, Xinit, yinit, noise_std=None):
    self.seq_len = seq_len
    self.noise_std = noise_std
    self.current_best_seq = None
    self.current_best_val = np.inf
    self.X_sample = []
    self.y_sample = []
    self.nqueries = 0
    m52 = ConstantKernel(1.0) * StringEmbedKernel(seq_len=seq_len, embedder=embedder)
    if noise_std is None:
      noise_std = np.std(yinit)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=noise_std**2)
    gpr.fit(Xinit, yinit)
    self.gpr = gpr
  
  def propose(self):
    self.nqueries += 1
    return gen_rnd_dna(self.seq_len)
 
  def update(self, x, y):
    if y > self.current_best_val:
      self.current_best_seq = x
      self.current_best_val = y
      
  def current_best(self):
    return (self.current_best_seq, self.current_best_val)
  

def embed(x):
  return embedder.predict(x)


    
nsteps = 100
solver = GPMaximizer(seq_len, embed, Xinit, yinit)
history = []
for t in range(nsteps):
  x = solver.propose()
  y = oracle(x)
  print("queried {}, observed {}".format(x, y))
  solver.update(x, y)
  xbest, ybest = solver.current_best()
  history.append(ybest)
  
plt.figure()
plt.plot(range(nsteps), history)
plt.title('Enumerative solver')
      
      