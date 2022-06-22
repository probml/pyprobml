# Necessary functions for demo and ClassConditionalBMM
# Author : Aleyna Kara(@karalleyna)

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad
from jax.random import PRNGKey, split, permutation
from jax.lax import scan
from jax.scipy.special import expit, logit
from jax.experimental import optimizers

import distrax
from distrax._src.utils import jittable

from mixture_lib import MixtureSameFamily
import itertools

opt_init, opt_update, get_params = optimizers.adam(1e-1)

class ClassConditionalBMM(jittable.Jittable):
    def __init__(self, mixing_coeffs, probs, class_priors, n_char, threshold=1e-10):
        self.mixing_coeffs = mixing_coeffs
        self.probs = probs
        self.class_priors = class_priors
        self.model = (logit(mixing_coeffs), logit(probs))
        self.num_of_classes = n_char
        self.threshold = threshold
        self.log_threshold = jnp.log(threshold)

    @property
    def class_priors(self):
        return self._class_priors

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        mixing_coeffs_logits, probs_logits = value
        self._model = MixtureSameFamily(mixture_distribution=distrax.Categorical(logits=mixing_coeffs_logits),
                                        components_distribution=distrax.Independent(
                                            distrax.Bernoulli(logits=probs_logits),
                                            reinterpreted_batch_ndims=1))

    @class_priors.setter
    def class_priors(self, value):
        self._class_priors = distrax.Categorical(probs=value)

    def _cluster(self, observations, targets):
        '''
        Arranges the observations so that the number of the observations belonging
        to each class is equal and separates from each other.

        Parameters
        ----------
        observations : array
            Dataset

        targets : array
            The class labels of the given dataset

        Returns
        -------
        * array
            The new dataset that has one additional axis to separate the observations
            of different classes
        '''
        clusters = []
        min_n_sample = float('inf')

        for c in range(self.num_of_classes):
            obs_of_same_class = observations[jnp.nonzero(targets == c)]
            n_obs = obs_of_same_class.shape[0]
            min_n_sample = min(min_n_sample, n_obs)
            clusters.append(obs_of_same_class.reshape((n_obs, -1)))

        return jnp.vstack([obs_of_same_class[jnp.newaxis, 0:min_n_sample, :] for obs_of_same_class in clusters])

    @jit
    def loglikelihood(self, X, c):
        '''
        Calculates the log likelihood of the observations with its ground-truth class.

        Parameters
        ----------
        X : array
          The collection of data points that belongs to the same class

        c : int
          The ground-truth class

        Returns
        -------
        * array
            Log likelihoods found per data point
        '''

        mixing_loglikelihood = jnp.clip(self._model.mixture_distribution.logits[c], a_max=0, a_min=self.log_threshold)
        bern_loglikelihood_1 = X @ jnp.clip(self._model.components_distribution.distribution.logits[c].T, a_max=0,
                                            a_min=self.log_threshold)
        bern_loglikelihood_0 = (1 - X) @ jnp.clip(
            jnp.log(1 - jnp.clip(self._model.components_distribution.distribution.probs[c],
                                 a_min=self.threshold, a_max=1.0)), a_min=self.log_threshold, a_max=0).T
        mix_loglikelihood = mixing_loglikelihood + bern_loglikelihood_1 + bern_loglikelihood_0
        return mix_loglikelihood

    @jit
    def logsumexp(self, matrix, keepdims=True):
        '''
        Traditional logsumexp except constraining the values to be greater than
        a pre-determined lower bound.

        Parameters
        ----------
        matrix : array
        keepdims : bool

        Returns
        -------
        * array

        '''
        M = jnp.max(matrix, axis=-1)
        M = M[:, None]

        bern_sum = jnp.sum(jnp.exp(matrix - M), axis=1, keepdims=keepdims)
        bern_sum = jnp.where(bern_sum < self.threshold, self.threshold, bern_sum)
        log_bern_sum = M + jnp.log(bern_sum)
        return log_bern_sum

    def _sample_minibatches(self, iterables, batch_size):
        '''
        Creates mini-batches generator in which there are given number of elements 

        Parameters
        ----------
        iterables : array
            Data points with their class labels

        batch_size : int
            The number of observation sequences that will be included in
            each minibatch

        Returns
        -------
        * tuple
            Minibatches
        '''
        observations, targets = iterables
        N = len(observations)
        for idx in range(0, N, batch_size):
            yield observations[idx:min(idx + batch_size, N)], targets[idx:min(idx + batch_size, N)]

    @jit
    def expectation(self, X, c):
        '''
        E step

        Parameters
        ----------
        X : array
          The collection of data points that belongs to the same class

        c : int
          The ground-truth class

        Returns
        -------
        * array
            Gamma

        * array
           The mean of loglikelihoods with class priors
        '''
        mix_loglikelihood = self.loglikelihood(X, c)
        mix_loglikelihood_sum = self.logsumexp(mix_loglikelihood)
        gamma = jnp.exp(mix_loglikelihood - mix_loglikelihood_sum)
        gamma = jnp.where(gamma > self.threshold, gamma, self.threshold)
        return gamma, jnp.mean(mix_loglikelihood_sum)

    @jit
    def maximization(self, X, gamma):
        '''
        Maximization step

        Parameters
        ----------
        X : array
            Dataset

        Returns
        -------
        * array
            Mixing coefficients

        * array
           Probabilities
        '''
        gamma_sum = jnp.sum(gamma, axis=0, keepdims=True)

        # mu
        probs = gamma.T @ X / gamma_sum.T
        probs = jnp.where(probs < self.threshold, self.threshold, probs)
        probs = jnp.where(probs < 1, probs, 1 - self.threshold)

        # sigma
        mixing_coeffs = gamma_sum / X.shape[0]
        mixing_coeffs = jnp.where(mixing_coeffs < self.threshold, self.threshold, mixing_coeffs)
        mixing_coeffs = jnp.where(mixing_coeffs < 1, mixing_coeffs, 1 - self.threshold).squeeze()
        return mixing_coeffs, probs

    def fit_em(self, observations, targets, num_of_iters=10):
        '''
        Fits the model using em algorithm.

        Parameters
        ----------
        observations : array
            Dataset

        targets : array
            Ground-truth class labels

        num_of_iters : int
            The number of iterations the training process that takes place

        Returns
        -------
        * array
            Log likelihoods found per iteration
        '''
        iterations = jnp.arange(num_of_iters)
        classes = jnp.arange(self.num_of_classes)
        X = self._cluster(observations, targets)

        def train_step(params, i):
            self.model = params

            # Expectation
            gamma, log_likelihood = vmap(self.expectation, in_axes=(0, 0))(X, classes)

            # Maximization
            mixing_coeffs, probs = vmap(self.maximization, in_axes=(0, 0))(X, gamma)
            return (logit(mixing_coeffs), logit(probs)), -jnp.mean(log_likelihood)

        initial_params = (logit(self.mixing_coeffs),
                          logit(self.probs))

        final_params, history = scan(train_step, initial_params, iterations)
        self.model = final_params
        return history

    @jit
    def loss_fn(self, params, batch):
        '''
        Calculates expected mean negative loglikelihood.
        Parameters
        ----------
        params : tuple
            Consists of mixing coefficients and probabilities of the Bernoulli distribution respectively.
        batch : array
            The subset of observations with their targets

        Returns
        -------
        * int
            Negative log likelihood
        '''
        observations, targets = batch
        mixing_coeffs_logits, probs_logits = params
        self.model = (mixing_coeffs_logits, probs_logits)
        mix_loglikelihood = vmap(self.loglikelihood)(observations, targets)
        mix_loglikelihood_sum = jnp.nan_to_num(self.logsumexp(mix_loglikelihood), nan=self.log_threshold)
        return -jnp.mean(mix_loglikelihood_sum)

    @jit
    def update(self, i, opt_state, batch):
        '''
        Updates the optimizer state after taking derivative
        i : int
            The current iteration
        opt_state : jax.experimental.optimizers.OptimizerState
            The current state of the parameters
        batch : array
             The subset of observations with their targets

        Returns
        -------
        * jax.experimental.optimizers.OptimizerState
            The updated state
        * int
            Loss value calculated on the current batch
        '''
        params = get_params(opt_state)
        loss, grads = value_and_grad(self.loss_fn)(params, batch)
        return opt_update(i, grads, opt_state), loss

    def fit_sgd(self, observations, targets, batch_size, rng_key=None, optimizer=None, num_epochs=1):
        '''
        Fits the class conditional bernoulli mixture model using gradient descent algorithm with the given hyperparameters.
        Parameters
        ----------
        observations : array
            The observation sequences which Bernoulli Mixture Model is trained on
        targets : array
            The ground-truth classes
        batch_size : int
            The size of the batch
        rng_key : array
            Random key of shape (2,) and dtype uint32
        optimizer : jax.experimental.optimizers.Optimizer
            Optimizer to be used
        num_epochs : int
            The number of epoch the training process takes place

        Returns
        -------
        * array
            Mean loss values found per epoch
        '''
        global opt_init, opt_update, get_params

        if rng_key is None:
            rng_key = PRNGKey(0)

        if optimizer is not None:
            opt_init, opt_update, get_params = optimizer

        opt_state = opt_init((logit(self.mixing_coeffs), logit(self.probs)))
        itercount = itertools.count()

        num_complete_batches, leftover = jnp.divmod(num_epochs, batch_size)
        num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)

        def epoch_step(opt_state, key):
            perm = permutation(key, len(observations))
            _observatios, _targets = observations[perm], targets[perm]
            sample_generator = self._sample_minibatches((_observatios, _targets), batch_size)

            def train_step(opt_state, i):
                opt_state, loss = self.update(next(itercount), opt_state, next(sample_generator))
                return opt_state, loss

            opt_state, losses = scan(train_step, opt_state, jnp.arange(num_batches))
            return opt_state, losses.mean()

        epochs = split(rng_key, num_epochs)
        opt_state, history = scan(epoch_step, opt_state, epochs)
        params = get_params(opt_state)
        mixing_coeffs_logits, probs_logits = params

        self.model = (mixing_coeffs_logits, probs_logits)
        self.mixing_coeffs = expit(mixing_coeffs_logits)
        self.probs = expit(probs_logits)
        return history

    def predict(self, X):
        '''
        Predicts the class labels of the given observations

        Parameters
        ----------
        observations : array
            Dataset

        Returns
        -------
        * array
            Predicted classes

        * array
            Log likelihoods given class labels
        '''
        N, _ = X.shape
        classes = jnp.arange(self.num_of_classes)

        def ll(cls):
            mix_loglikelihood = self.loglikelihood(X, cls)
            sum_mix_loglikelihood = self.logsumexp(mix_loglikelihood)
            bayes = self.class_priors.logits[..., cls] + sum_mix_loglikelihood
            return bayes.flatten()

        ll_given_c = vmap(ll, out_axes=(1))(classes)
        predictions = jnp.argmax(ll_given_c, axis=-1)

        return predictions, ll_given_c