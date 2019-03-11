import numpy as np
import pandas as pd
import utils

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
  y = utils.zscore_normalize(y)
  return X, y

def tfbind_problem(max_ntrain = None, lower_bin=40, upper_bin=60):
  Xall, yall = get_8mer_data()  
  # Extract training set based on "medium performing" strings
  bins = pd.qcut(yall, 100, labels=False, duplicates='drop')
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
  
  return oracle, Xall, yall, Xtrain, ytrain, middle_bins

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

  Xall = utils.gen_all_strings(seq_len) # (N,L) array of ints (in 0..A)
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
  

  return oracle, Xall, yall, Xtrain, ytrain, middle_bins

