import superimport

import collections
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import pyprobml_utils as pml

    
data_dir = "../data"
fname = os.path.join(data_dir, 'timemachine.txt')

with open(fname, 'r') as f:
    lines = f.readlines()
    raw_dataset = [re.sub('[^A-Za-z]+', ' ', st).lower().split()
                   for st in lines]

# Print first few lines
for sentence in raw_dataset[:10]:
    print(sentence)
    
# Concat sentences into single string of chars
#  skip blank lines
sentences = [' '.join(s) for s in raw_dataset if s]

# concat into single long string
charseq = ''.join(sentences)


# Unigrams
wseq = charseq
print('First 10 unigrams\n', wseq[:10])

# Bigrams
word_pairs = [pair for pair in zip(wseq[:-1], wseq[1:])]
print('First 10 bigrams\n', word_pairs[:10])

# Trigrams
word_triples = [triple for triple in zip(wseq[:-2], wseq[1:-1], wseq[2:])]
print('First 10 trigrams\n', word_triples[:10])

# ngram statistics
counter = collections.Counter(wseq)
counter_pairs = collections.Counter(word_pairs)
counter_triples = collections.Counter(word_triples)

print('Most common unigrams\n', counter.most_common(10))
print('Most common bigrams\n', counter_pairs.most_common(10))
print('Most common trigrams\n', counter_triples.most_common(10))


# convert [(('t', 'h', 'e'), 3126), ...] to {'the': 3126, ...}
def make_dict(lst, min_count=1):
    d = dict()
    for s, c in lst:
        if c <= min_count:
            continue
        key = ''.join(s)
        d[key] = c
    return d

unigram_dict = make_dict(counter.most_common())
alphabet = list(unigram_dict.keys())
alpha_size = len(alphabet)

trigram_dict = make_dict(counter_triples.most_common())

# convert [(('t', 'h', 'e'), 3126), ( ('t', 'h', 'o'), 444)] to {'th': {'e': 3126, 'o': 444) }...}
def make_cond_dict(lst, min_count=1):
    d = dict()
    for s, c in lst:
        if c <= min_count:
            continue
        history = ''.join(s[:-1])
        future = s[-1]
        future_counts = d.get(history, {})
        future_counts[future] = c
        d[history] = future_counts
    return d

lst = [(('t', 'h', 'e'), 3126), ( ('t', 'h', 'o'), 444), (('b', 'o'), 22)]
d = make_cond_dict(lst)

trigram_cond_dict = make_cond_dict(counter_triples.most_common())


'''
def gen_rnd_string(seq_len, alphabet):
  s = [random.choice(alphabet) for i in range(seq_len)]
  return ''.join(s)

def gen_all_strings(seq_len, alphabet):
  S = [np.array(p) for p in itertools.product(alphabet, repeat=seq_len)]
  return np.stack(S)


def encode_strings(S, alphabet):
  # S is an N-list of L-strings, L=8, N=65536
  def encode_char(c):
      return alphabet.index(c)
  S1 = [list(s) for s in S] # N-list of L-lists
  S2 = np.array(S1) # (N,L) array of strings, N=A**L
  X = np.vectorize(encode_char)(S2) # (N,L) array of ints (in 0..A)
  return X
'''

history = 'ab'
T = 10
cond_dict = trigram_cond_dict
for t in range(T):
    counts_local = cond_dict[history]

    #counts for possible continuations given the previous order characters
    list_local = [(k,counts_local[k]) for k in counts_local]

    #probability norm == count sum
    total_local = np.sum([ c for (s,c) in list_local ])
    probs = [float(c)/total_local for (s,c) in list_local] #probabilities

    cum_probs = [] #cummulative probabilities
    p_sum = 0.0
    for p in probs:
        p_sum += p
        cum_probs.append(p_sum)

    #choosing an option from a list with probs as probabilities
    p_now = np.random.rand()
    cond = (np.array(cum_probs) > p_now)
    ids = [(i-1) for i,x in enumerate(cond) if x == True]
    choice_id = min(ids)

    next_char = list_local[choice_id][0]

    text = text + next_char
    #print(t,order)

print(text)

