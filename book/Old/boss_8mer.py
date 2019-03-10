
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import zscore_normalize
import boss_utils

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
  ndx = np.where((Xall==x).all(axis=1))[0][0]
  return yall[ndx]

def oracle_batch(X):
  return np.apply_along_axis(oracle, 1,  X)


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

predictor = boss_utils.learn_supervised_model(Xtrain, ytrain)
ypred = predictor.predict(Xall)
plt.figure()
plt.scatter(yall, ypred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

embedder = boss_utils.convert_to_embedder(predictor, seq_len)

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

def embed_fn(x):  
  return embedder.predict(x)  
  
n_iter=10
methods = []
methods.append('enum')
methods.append('bayes')
for method in methods:
  np.random.seed(0)
  ytrace = boss_utils.boss_maximize(method, oracle, Xinit, yinit,  embed_fn, n_iter=n_iter)    
  plt.figure()
  plt.plot(ytrace)
  plt.title(method)


  