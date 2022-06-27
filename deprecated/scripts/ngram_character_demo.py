'''
This script consists of functions that allow users to fit ngram model, sample from an ngram model and calculate the
log likelihood of the given sequence given an ngram model.

Author : Aleyna Kara(@karalleyna)
'''
import superimport
from nltk.util import ngrams
from nltk import FreqDist, LidstoneProbDist
from dataclasses import dataclass
from collections import defaultdict
import re
import string
import numpy as np
import requests

@dataclass
class NGram:
    freq_dist: FreqDist
    prob_dists: defaultdict
    N: int


def preprocessing(text, case_folding=False):
    preprocessed_text = text.lower() if case_folding else text
    preprocessed_text = re.sub(r'\d+', '', preprocessed_text)
    preprocessed_text = preprocessed_text.translate(str.maketrans('', '', string.punctuation + "’↵·"))
    preprocessed_text = re.sub(r'\s+', ' ', preprocessed_text)
    return preprocessed_text


def read_file(filepath):
    f = open(filepath, 'r')
    text = f.read()
    return text


def ngram_model_fit(n, data, smoothing=1):
    n_grams = ngrams(data, n)
    ngram_fd = FreqDist(n_grams)
    probs_dists = get_probs_dist(ngram_fd, smoothing)
    model = NGram(ngram_fd, probs_dists, n)
    return model


def get_probs_dist(freq_dist, smoothing=1):
    freq_dists = defaultdict(FreqDist)

    for ngram, freq in freq_dist.items():
        *prefix, cur = ngram
        key = ''.join(prefix)
        freq_dists[key].update({cur: freq_dist[ngram]})

    probs_dist = defaultdict(LidstoneProbDist)

    for prefix, fd in freq_dists.items():
        probs_dist[prefix] = LidstoneProbDist(fd, gamma=smoothing)

    return probs_dist


def ngram_model_sample(model, text_length, prefix, seed=0):
    assert len(prefix) >= model.N - 1
    np.random.seed(seed)
    text = prefix
    for _ in range(text_length):
        cur_prefix = text[-1 * model.N + 1:]
        if cur_prefix not in model.prob_dists:
            return text
        new_char = model.prob_dists[cur_prefix].generate()
        text = text + new_char
    return text


def ngram_loglikelihood(model, seq):
    assert len(seq) >= model.N - 1

    prefix = seq[:model.N - 1]
    if prefix not in model.prob_dists:
        return float("-inf")
    ll = np.log(model.prob_dists[prefix].freqdist().N() / model.freq_dist.N())

    for i in range(model.N - 1, len(seq)):
        prev = seq[i - model.N + 1: i]
        cur = seq[i]

        if prev not in model.prob_dists:
            return float("-inf")

        ll += model.prob_dists[prev].logprob(cur)

    return ll


url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/bible.txt'
response = requests.get(url)
text = response.content.decode("utf-8")
data = preprocessing(text, case_folding=True)

n = 10
model = ngram_model_fit(n, data, smoothing=1)

sample = ngram_model_sample(model, text_length=500, prefix='christian', seed=0)
print(sample)

# ngram statistics
#print(f'Most common {n}-grams\n', model.freq_dist.most_common(10))


