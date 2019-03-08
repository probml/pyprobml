import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_SITES = ['HOXC4_REF_R2', 'CRX_REF_R1']

def load_data(file_name, normalize=False):
  #Function to load the data source file 
  data = pd.read_csv(file_name, sep=',')
  data = data.iloc[:, 1:]  # Remove index column
  assert data['seq'].nunique() == len(data)
  data = data.set_index('seq')
  if normalize:
    data = (data - data.min()) / np.maximum(data.max() - data.min(), 1e-8)
  data = data.astype(np.float32)
  return data


def zscore_normalize(data):
  return (data - data.mean()) / np.maximum(data.std(), 1e-8)

def min_max_normalize(data):
  return (data - data.min()) / np.maximum(data.max() - data.min(), 1e-8)

"""
data_path = '/home/kpmurphy/github/pyprobml/data/8mer.csv'
data = load_data(data_path)
datan = min_max_normalize(data)

ndx=np.where(datan.columns == DEFAULT_SITES[0])
dat = datan[DEFAULT_SITES[0]]
"""

file_name = '/home/kpmurphy/github/pyprobml/data/8mers_crx_ref_r1.csv'
data = pd.read_csv(file_name, sep='\t')
S8 = data['seq'].values
y8 = data['val'].values

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

X8 = encode_data(S8)
y8 = zscore_normalize(data['val'])

plt.figure(figsize=(10, 4))
plt.plot(y8)
plt.title('8mer scores')
plt.show()

bins = pd.qcut(y8, 1000, labels=False, duplicates='drop')
max_bin = np.max(bins)
tops = np.where(bins == max_bin)[0]
bots = np.where(bins <= 1)[0]
print(np.mean(y8[tops]))

ndx=np.where(y8>=14)[0];np.size(ndx)
 
