

import tensorflow as tf
from tensorflow import keras
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(0)

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


seq_len = 8 # L
alpha_size = 4 # A
nseq = alpha_size ** seq_len
print("Generating {} sequences of length {}".format(nseq, seq_len))

def gen_rnd_dna(seq_len):
  s = [random.choice([0,1,2,3]) for i in range(seq_len)]
  return np.array(s)

def gen_all_dna(seq_len):
  S = [np.array(p) for p in itertools.product([0,1,2,3], repeat=seq_len)]
  return np.stack(S)



  
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
