'''
Implements the class-conditional HMM where the class label specifies one of C possible words, chosen from some vocabulary.
If the class is a word of length T, it has a deterministic left-to-right state transition matrix A
with T possible states.The t'th state should have an categorical emission distribution which generates t'th letter
in lower case with probability p1, in upper case with probability p1, a blank character "-" with prob p2,
and a random letter with prob p3.
Author : Aleyna Kara (@karalleyna)
'''

import superimport

import numpy as np
import matplotlib.pyplot as plt
from hmm_lib import HMMDiscrete

class Word:
    '''
    This class consists of components needed for a class-conditional Hidden Markov Model
    with categorical distribution

    Parameters
    ----------
    word: str
      Class label representing a word
    p1: float
      The probability of the uppercase and lowercase letter included within
      a word for the current state
    p2: float
      The probability of the blank character
    p3: float
      The probability of the uppercase and lowercase letters except correct one
    L : int
      The number of letters used when constructing words
    type_ : str
      "all" : Includes both uppercase and lowercase letters
      "lower" : Includes only lowercase letters
      "upper" : Includes only uppercase letters
    '''

    def __init__(self, word, p1, p2, p3, L, type_):
        self.word, self.T = word, len(word)
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.type_ = type_
        self.L = 2 * L if self.type_ == 'all' else L
        self.init_state_dist = np.zeros((self.T + 1,))
        self.init_state_dist[0] = 1
        self.init_state_transition_matrix()
        self.init_emission_probs()

    def init_state_transition_matrix(self):
        assert self.T > 0
        A = np.zeros((self.T + 1, self.T + 1))  # transition-probability matrix
        A[:-1, 1:] = np.eye(self.T)
        A[-1, 0] = 1
        self.A = A

    def emission_prob_(self, letter):
        ascii_no = ord(letter.upper()) - 65  # 65 :ascii number of A
        idx = [ascii_no, ascii_no + self.L // 2] if self.type_ == 'all' else ascii_no
        emission_prob = np.full((1, self.L), self.p3)
        emission_prob[:, idx] = self.p1
        return emission_prob

    def init_emission_probs(self):
        self.B = np.zeros((self.T, self.L))  # observation likelihoods
        for i in range(self.T):
            self.B[i] = self.emission_prob_(self.word[i])
        self.B = np.c_[self.B, np.full((self.T, 1), self.p2), np.zeros((self.T, 1))]
        self.B = np.r_[self.B, np.zeros((1, self.L + 2))]
        self.B[-1, -1] = 1

    def sample(self, n_word, random_states=None):
        '''
        n_word: int
          The number of times sampled a word by HMMDiscrete
        random_states: List[int]
          The random states each of which is given to HMMDiscrete.sample as a parameter
        '''
        random_states = np.random.randint(0, 2 * n_word, n_word) if random_states is None or n_word > len(
            random_states) else random_states
        self.hmm_discrete = HMMDiscrete(self.A, self.B, self.init_state_dist)
        return np.r_[[self.hmm_discrete.sample(self.T + 1, rand_stat)[1] for rand_stat in random_states]]

    def likelihood(self, word, cls_prob):
        '''
        Calculates the class conditional likelihood for a particular word with
        given class probability by using  p(word|c) p(c) where c is the class itself
        '''
        return cls_prob * np.exp(self.hmm_discrete.forwards(word)[1])

    def loglikelihood(self, X):
        return np.array([self.hmm_discrete.forwards(x)[1] for x in X])

# converts the list of integers into a word
def decode(idx, L, type_):
  if idx==L+2:
    return ' '
  elif idx==L:
    return '-'
  elif type_=="all" and idx >= L // 2:
    return chr(idx - L // 2 + 65)
  elif type_=="upper":
    return chr(idx + 65)
  return chr(idx + 97)

# encodes the word into list of integers
def encode(word, L , type_):
  arr = np.array([], dtype=int)
  for c in word:
    if c=='-':
      arr = np.append(arr, [2*L if type_=='all' else L])
    elif type_ == 'all':
      arr = np.append(arr, [ord(c)-39 if c.isupper() else ord(c)-97]) # ascii no of A : 65, ascii no of a : 97
    else:
      arr = np.append(arr, [ord(c.upper())-65])
  return arr


L, T = 26, 4
p1, p2, p3 = 0.4, 0.1, 2e-3

vocab = ['book', 'bird', 'bond', 'bone', 'bank', 'byte', 'pond', 'mind', 'song', 'band']
hmms = {word: Word(word, p1, p2, p3, L, "all") for word in vocab}

n_misspelled = 5  # number of misspelled words created for each class
random_states = np.arange(0, len(vocab) * n_misspelled).reshape((-1, n_misspelled))

samples = [hmms[vocab[i]].sample(n_misspelled, random_states[i]) for i in range(len(vocab))]
misspelled_words = [''.join([decode(letter, L, "all") for letter in word][:-1]) for sample in samples for word in
                    sample]

# noisy words
test = ['bo--', '-On-', 'b-N-', 'B---', '-OnD', 'b--D', '---D', '--Nd', 'B-nD', '-O--', 'b--d', '--n-']
p_c = 1 / len(vocab)
fig, axes = plt.subplots(4, 3, figsize=(20, 10))

for i, (ax, word) in enumerate(zip(axes.flat, test)):
    ax.bar(vocab, [hmms[w].likelihood(encode(word, L, 'all'), p_c) for w in vocab])
    ax.set_title(f'{word}')
plt.tight_layout()
plt.show()

