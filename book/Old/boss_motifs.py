
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import boss_utils
import utils

np.random.seed(0)

def motif_distance(x, m):
  # hamming distance of x to motif
  # If m[i]=nan, it means locn i is a don't care
  mask = [not(np.isnan(v)) for v in m] #np.where(m>0)
  return np.sum(x[mask] != m[mask])

seq_len = 3 # L
alpha_size = 4 # A
nseq = alpha_size ** seq_len
print("Generating {} sequences of length {}".format(nseq, seq_len))

motifs = [];
#m = np.arange(seq_len, dtype=float)
m = np.repeat(3.0, seq_len)
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
  d = seq_len - d # closer implies higher score
  d = d + np.random.normal(0, 0.01)
  return d

def oracle_batch(X):
  return np.apply_along_axis(oracle, 1,  X)


Xall = utils.gen_all_strings(seq_len) # (N,L) array of ints (in 0..A)
yall = oracle_batch(Xall)

plt.figure()
plt.plot(yall)

Xtrain = Xall
ytrain = yall

predictor = boss_utils.learn_supervised_model(Xtrain, ytrain)
ypred = predictor.predict(Xall)
plt.figure()
plt.scatter(yall, ypred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

embedder = boss_utils.convert_to_embedder(predictor, seq_len)
def embed_fn(x):  
  return embedder.predict(x)  
  
Xinit = Xtrain[:10]
yinit = ytrain[:10]
n_iter=2
methods = []
methods.append('bayes')
methods.append('random')
for method in methods:
  np.random.seed(0)
  ytrace = boss_utils.boss_maximize(method, oracle, Xinit, yinit,  embed_fn, n_iter=n_iter)    
  plt.figure()
  plt.plot(ytrace)
  plt.title(method)