import superimport

import collections
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml
import requests


url  = 'https://raw.githubusercontent.com/probml/probml-data/main/data/timemachine.txt'
response = requests.get(url)
data = response.text
lines = [s+'\n' for s in response.text.split("\n")]
raw_dataset = [re.sub('[^A-Za-z]+', ' ', st).lower().split() for st in lines]

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

bigram_dict = make_dict(counter_pairs.most_common())

bigram_count = np.zeros((alpha_size, alpha_size))
for k, v in bigram_dict.items():
    code0 = alphabet.index(k[0])
    code1 = alphabet.index(k[1])
    #print('code0 {}, code1 {}, k {}, v {}'.format(code0, code1, k, v))
    bigram_count[code0, code1] += v
    
bigram_prob = bigram_count / (1e-10+np.sum(bigram_count,axis=1))

#https://matplotlib.org/3.1.1/gallery/specialty_plots/hinton_demo.html
def hinton_diagram(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    ax.axis('on')
    ax.set_xticks(range(alpha_size))
    ax.set_xticklabels(alphabet)
    ax.set_yticks(range(alpha_size))
    ax.set_yticklabels(alphabet)

    
plt.figure(figsize=(8,8))
hinton_diagram(bigram_count.T)
pml.savefig('bigram-count.pdf')
plt.show()

plt.figure(figsize=(8,8))
hinton_diagram(bigram_prob.T)
pml.savefig('bigram-prob.pdf')
plt.show()

