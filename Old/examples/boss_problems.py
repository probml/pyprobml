import numpy as np
import pandas as pd


np.random.seed(0)
import random
import itertools

def gen_rnd_string(seq_len, alphabet=[0,1,2,3]):
  s = [random.choice(alphabet) for i in range(seq_len)]
  return np.array(s)

def gen_all_strings(seq_len, alphabet=[0,1,2,3]):
  S = [np.array(p) for p in itertools.product(alphabet, repeat=seq_len)]
  return np.stack(S)

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
    # There are (4**8 - 256)/2 rows
    # 256 is the number of palindromes 
    # We divide by 2 since the DNA is double stranded
    fname = 'CRX_REF_R1_8mers.txt'
    import os
    import pandas as pd
    path = os.environ["PYPROBML"]
    datadir = os.path.join(path, "data")  
    fullname = os.path.join(datadir, fname)
    data = pd.read_csv(fullname, sep='\t') 
    S = data['8-mer'].values
    y = data['Z-score'].values
    X = encode_data(S)
    #y = zscore_normalize(y)
    return X, y

def tfbind_problem(max_ntrain = None, lower_bin=40, upper_bin=60, max_nseq=None):
  Xall, yall = get_8mer_data()  
  if max_nseq is not None:
    nseq = np.shape(Xall)[0]
    perm = np.random.permutation(nseq)
    Xall = Xall[perm[:max_nseq]]
    yall = yall[perm[:max_nseq]]
  # Extract training set based on "medium performing" strings
  bins = pd.qcut(yall, 100, labels=False, duplicates='drop')
  print("tfbind bins min {} max {}".format(min(bins), max(bins)))
  middle_bins = np.where(np.logical_and(bins>=lower_bin, bins<=upper_bin))[0]
  Xtrain = Xall[middle_bins]
  ytrain = yall[middle_bins]
  ntrain = np.shape(Xtrain)[0]
  if max_ntrain is not None:
    if ntrain > max_ntrain:
      perm = np.random.permutation(ntrain)
      Xtrain = Xtrain[perm[:max_ntrain]]
      ytrain = ytrain[perm[:max_ntrain]]
      ntrain = max_ntrain
  nseq = np.shape(Xall)[0]
  print("TFbind: Choosing {} training examples from {}".format(ntrain, nseq))
  
  def oracle(x):
    ndx = np.where((Xall==x).all(axis=1))[0][0]
    return yall[ndx]
  
  # https://stackoverflow.com/questions/2063425/python-elegant-inverse-function-of-intstring-base
  stringify = lambda x: ''.join([str(digit) for digit in x])
  Sall = np.apply_along_axis(stringify, 1, Xall)
  Nall = np.array([int(s,4) for s in Sall])

  def oracle_batch(X):
    S = np.apply_along_axis(stringify, 1, X)
    encoded = [int(s,4) for s in S]
    ndx = [np.where(n == Nall)[0][0] for n in encoded]
    return yall[ndx]
  
  return oracle, oracle_batch, Xall, yall, Xtrain, ytrain, middle_bins

###################

  
def motif_distance(x, m):
  # hamming distance of x to motif
  # If m[i]=nan, it means locn i is a don't care
  mask = [not(np.isnan(v)) for v in m] #np.where(m>0)
  return np.sum(x[mask] != m[mask])

def make_motifs(seq_len):
  motifs = [];
  m = np.repeat(3.0, seq_len)
  m1 = np.copy(m)
  m1[0] = np.nan
  m2 = np.copy(m)
  m2[seq_len-1] = np.nan
  motifs = [m1, m2]
  #motifs = [m2]
  return motifs


def motif_problem(seq_len, noise=0, lower_bin=40, upper_bin=70):
  motifs = make_motifs(seq_len)
  
  def oracle(x):
    d = np.inf
    for motif in motifs:
      d = min(d, motif_distance(x, motif))
    # Max distance is L, min is 0
    # Since we want to maximize the objective, we return L-d
    d = seq_len - d 
    if noise > 0:
      d = d + np.random.normal(0, noise)
    return d
  
  def oracle_batch(X):
    return np.apply_along_axis(oracle, 1,  X)

  Xall = gen_all_strings(seq_len) # (N,L) array of ints (in 0..A)
  yall = oracle_batch(Xall)

  # Extract training set based on "medium performing" strings
  #bins = pd.qcut(yall, 10, labels=False, duplicates='drop') #0..9
  #middle_bins = np.where(np.logical_and(bins>=4, bins<=7))[0]
  bins = pd.qcut(yall, 100, labels=False, duplicates='drop')
  middle_bins = np.where(np.logical_and(bins>=lower_bin, bins<=upper_bin))[0]
  Xtrain = Xall[middle_bins]
  ytrain = yall[middle_bins]
  ntrain = np.shape(Xtrain)[0]
  nseq = np.shape(Xall)[0]
  print("Motifs: Choosing {} training examples from {}".format(ntrain, nseq))
  

  return oracle, oracle_batch, Xall, yall, Xtrain, ytrain, middle_bins

