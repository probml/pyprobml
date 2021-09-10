'''
Implements the class-conditional HMM where the class label specifies one of C possible words, chosen from some vocabulary.
If the class is a word of length T, it has a deterministic left-to-right state transition matrix A
with T possible states.The t'th state should have an categorical emission distribution which generates t'th letter
in lower case with probability p1, in upper case with probability p1, a blank character "-" with prob p2,
and a random letter with prob p3.
Author : Aleyna Kara (@karalleyna)
'''

#!pip install -q distrax
import distrax

from jax import vmap, jit
from jax.random import PRNGKey, split, uniform
import jax.numpy as jnp

from conditional_bernoulli_mix_lib import ClassConditionalBMM
from conditional_bernoulli_mix_utils import encode

from distrax._src.utils import jittable

from functools import partial
import numpy as np

class Word(jittable.Jittable):
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
    n_char : int
      The number of letters used when constructing words
    type : str
      "all" : Includes both uppercase and lowercase letters
      "lower" : Includes only lowercase letters
      "upper" : Includes only uppercase letters
    dataset : array
        Dataset
    targets : array
        The ground-truth labels of the dataset
    mixing_coeffs : array
        Mixing coefficients
    initial_probs : array
        Initial probabilities
    n_mix : int
        Number of component distributions
    rng_key : array
        Random key of shape (2,) and dtype uint32
    '''

    def __init__(self, word, p1, p2, p3, n_char, type, dataset=None, targets=None, mixing_coeffs=None, initial_probs=None,
                 n_mix=3, rng_key=None):
        self.word, self.word_len = word, len(word)
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.type_ = type
        self.n_char = n_char + 1 # num_of_letters + blank character
        self.n_mix = n_mix

        self.init_dist = None
        self.trans_dist = None
        self.init_emission_probs(mixing_coeffs, initial_probs, dataset, targets, rng_key=rng_key)

    @property
    def obs_dist(self):
        return self._obs_dist

    @property
    def init_dist(self):
        return self._init_dist

    @property
    def trans_dist(self):
        return self._trans_dist

    @trans_dist.setter
    def trans_dist(self, value=None):
        assert self.word_len > 0

        if value is None:
            value = jnp.eye(self.word_len)  # transition-probability matrix
            value = jnp.roll(value, 1, axis=1)
        self._trans_dist = distrax.Categorical(probs=value)

    @init_dist.setter
    def init_dist(self, value=None):
        if value is None:
            value = jnp.append(jnp.ones((1,)), jnp.zeros((self.word_len - 1,)))
        self._init_dist = distrax.Categorical(probs=value)

    def emission_prob_(self, letter):
        """
        Initializes emission probabilities for a given state
        Parameters
        ----------
        letter : str
            String of unit length, i.e. a character
        Returns
        -------
        * array
            Emission probabilities
        """
        ascii_no = ord(letter.upper()) - 65  # 65 :ascii number of A
        idx = [ascii_no, ascii_no + self.n_char // 2] if self.type_ == 'all' else (
            ascii_no if self.type_ == "upper" else ascii_no + self.n_char // 2)
        emission_prob = np.full((1, self.n_char), self.p3)
        emission_prob[:, -1] = self.p2

        if letter == "-":
            return emission_prob

        emission_prob[:, idx] = self.p1
        if self.type_ is not 'all':
            start = self.n_char // 2 if self.type_ is 'upper' else 0
            emission_prob[:, start: start + self.n_char // 2] = 0
        return emission_prob

    def init_emission_probs(self, mixing_coeffs, probs, dataset, targets, rng_key=None, num_of_iter=7):
        """

        Parameters
        ----------
        mixing_coeffs : array
            The probabilities of mixture_distribution of ClassConditionalBMM
        probs : array
            The probabilities of components_distribution of ClassConditionalBMM
        dataset : array
            Dataset
        targets :
            The ground-truth labels of the dataset
        rng_key : array
            Random key of shape (2,) and dtype uint32
        num_of_iter
            The number of iterations the training process that takes place
        Returns
        -------

        """
        class_priors = np.zeros((self.word_len, self.n_char))  # observation likelihoods

        for i in range(self.word_len):
            class_priors[i] = self.emission_prob_(self.word[i])

        if (mixing_coeffs is None or probs is None) and (dataset is not None and targets is not None):
            mixing_coeffs = jnp.full((self.n_char - 1, self.n_mix), 1. / self.n_mix)

            if rng_key is None:
                rng_key = PRNGKey(0)

            probs = uniform(rng_key, minval=0.4, maxval=0.6, shape=(self.n_char - 1, self.n_mix, dataset.shape[-1]))

            class_conditional_bmm = ClassConditionalBMM(mixing_coeffs, probs, jnp.array(class_priors), self.n_char - 1)

            class_conditional_bmm.fit_em(dataset, targets, num_of_iter)
            self._obs_dist = class_conditional_bmm

        else:
            self._obs_dist = ClassConditionalBMM(mixing_coeffs, probs, jnp.array(class_priors), self.n_char - 1)

    @jit
    def sample(self, rng_key):
        """

        Parameters
        ----------
        rng_key

        Returns
        -------

        """
        sample_key, rng_key = split(rng_key)
        keys = split(rng_key, self.word_len)
        class_priors = self._obs_dist.class_priors.sample(seed=sample_key)  # 4,

        def _sample(cls, key):
            return jnp.where(cls == self.n_char - 1, 0, self._obs_dist.model.sample(seed=key)[cls])

        obs_seq = vmap(_sample, in_axes=(0, 0))(class_priors, keys)
        return jnp.hstack([obs_seq, class_priors.reshape((-1, 1))])

    @partial(jit, static_argnums=(1,))
    def n_sample(self, n_misspelled_words, rng_key=None):
        """

        Parameters
        ----------
        n_misspelled_words: int
          The number of times sampled a word by HMM
        rng_key: array
          The PRNGKey each of which is splitted and then is given to HMM.sample as a parameter

        Returns
        -------

        """
        if rng_key is None:
            rng_key = PRNGKey(0)
        keys = split(rng_key, n_misspelled_words)

        misspelled = vmap(self.sample)(keys)
        return misspelled

    @partial(jit, static_argnums=(1,))
    def loglikelihood(self, word, images, log_threshold=-12):
        '''
        Calculates the class conditional likelihood for a particular word with
        given class probability by using  p(word|c) p(c) where c is the class itself
        '''
        n_pixels = images.shape[-1]

        def _loglikelihood(i, cls, img):
            prior = self._obs_dist.class_priors.logits[i, cls]
            logbern = jnp.where(cls == self.n_char - 1, jnp.ones((self.n_mix,)) * log_threshold * n_pixels,
                                self._obs_dist.loglikelihood(img, cls))
            return logbern, prior

        assert len(word) <= len(self.word)

        extended_word = word + "-" * (len(self.word) - len(word))
        images = jnp.vstack([images, jnp.zeros((len(self.word) - len(word), n_pixels))])
        classes = encode(extended_word, self.n_char, self.type_)
        indices = jnp.arange(len(self.word))

        ll, priors = vmap(_loglikelihood, in_axes=(0, 0, 0))(indices, classes, images)
        mix_loglikelihood_sum = self._obs_dist.logsumexp(ll)
        return mix_loglikelihood_sum + priors