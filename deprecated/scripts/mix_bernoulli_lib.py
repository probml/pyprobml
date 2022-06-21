# Implementation of Bernoulli Mixture Model
# Author : Aleyna Kara(@karalleyna)

import superimport

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad
from jax.random import PRNGKey, uniform, split, permutation
from jax.lax import scan
from jax.scipy.special import expit, logit
from jax.nn import softmax
from jax.experimental import optimizers

import distrax
from distrax._src.utils import jittable

from mixture_lib import MixtureSameFamily
import pyprobml_utils as pml

import matplotlib.pyplot as plt
import itertools

opt_init, opt_update, get_params = optimizers.adam(1e-1)

class BMM(jittable.Jittable):
    def __init__(self, K, n_vars, rng_key=None):
        '''
        Initializes Bernoulli Mixture Model

        Parameters
        ----------
        K : int
            Number of latent variables

        n_vars : int
            Dimension of binary random variable

        rng_key : array
            Random key of shape (2,) and dtype uint32
        '''
        rng_key = PRNGKey(0) if rng_key is None else rng_key

        mixing_coeffs = uniform(rng_key, (K,), minval=100, maxval=200)
        mixing_coeffs = mixing_coeffs / mixing_coeffs.sum()
        initial_probs = jnp.full((K, n_vars), 1.0 / K)

        self._probs = initial_probs
        self.model = (mixing_coeffs, initial_probs)

    @property
    def mixing_coeffs(self):
        return self._model.mixture_distribution.probs

    @property
    def probs(self):
        return self._probs

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        mixing_coeffs, probs = value
        self._model = MixtureSameFamily(mixture_distribution=distrax.Categorical(probs=mixing_coeffs),
                                        components_distribution=distrax.Independent(distrax.Bernoulli(probs=probs)))

    def responsibilities(self, observations):
        '''
         Finds responsibilities

         Parameters
         ----------
         observations : array(N, seq_len)
             Dataset

         Returns
         -------
         * array
             Responsibilities
         '''
        return jnp.nan_to_num(self._model.posterior_marginal(observations).probs)

    def expected_log_likelihood(self, observations):
        '''
        Calculates expected log likelihood

        Parameters
        ----------
        observations : array(N, seq_len)
            Dataset

        Returns
        -------
        * int
            Log likelihood
        '''
        return jnp.sum(jnp.nan_to_num(self._model.log_prob(observations)))

    def _m_step(self, observations):
        '''
        Maximization step

        Parameters
        ----------
        observations : array(N, seq_len)
            Dataset

        Returns
        -------
        * array
            Mixing coefficients

        * array
           Probabilities
        '''
        n_obs, _ = observations.shape

        # Computes responsibilities, or posterior probability p(z|x)
        def m_step_per_bernoulli(responsibility):
            norm_const = responsibility.sum()
            mu = jnp.sum(responsibility[:, None] * observations, axis=0) / norm_const
            return mu, norm_const

        mus, ns = vmap(m_step_per_bernoulli, in_axes=(1))(self.responsibilities(observations))
        return ns / n_obs, mus

    def fit_em(self, observations, num_of_iters=10):
        '''
        Fits the model using em algorithm.

        Parameters
        ----------
        observations : array(N, seq_len)
            Dataset

        num_of_iters : int
            The number of iterations the training process takes place

        Returns
        -------
        * array
            Log likelihoods found per iteration

        * array
            Responsibilities
        '''
        iterations = jnp.arange(num_of_iters)

        def train_step(params, i):
            self.model = params

            log_likelihood = self.expected_log_likelihood(observations)
            responsibilities = self.responsibilities(observations)

            mixing_coeffs, probs = self._m_step(observations)

            return (mixing_coeffs, probs), (log_likelihood, responsibilities)

        initial_params = (self.mixing_coeffs,
                          self.probs)

        final_params, history = scan(train_step, initial_params, iterations)

        self.model = final_params
        _, probs = final_params
        self._probs = probs

        ll_hist, responsibility_hist = history

        ll_hist = jnp.append(ll_hist, self.expected_log_likelihood(observations))
        responsibility_hist = jnp.vstack([responsibility_hist,
                                          jnp.array([self.responsibilities(observations)])])

        return ll_hist, responsibility_hist

    def _make_minibatches(self, observations, batch_size, rng_key):
        '''
        Creates minibatches consists of the random permutations of the
        given observation sequences

        Parameters
        ----------
        observations : array(N, seq_len)
            Dataset

        batch_size : int
            The number of observation sequences that will be included in
            each minibatch

        rng_key : array
            Random key of shape (2,) and dtype uint32

        Returns
        -------
        * array(num_batches, batch_size, max_len)
            Minibatches
        '''
        num_train = len(observations)
        perm = permutation(rng_key, num_train)

        def create_mini_batch(batch_idx):
            return observations[batch_idx]

        num_batches = num_train // batch_size
        batch_indices = perm.reshape((num_batches, -1))
        minibatches = vmap(create_mini_batch)(batch_indices)

        return minibatches

    @jit
    def loss_fn(self, params, batch):
        '''
        Calculates expected mean negative loglikelihood.

        Parameters
        ----------
        params : tuple
            Consists of mixing coefficients and probabilities of the Bernoulli distribution respectively.

        batch : array
            The subset of observations

        Returns
        -------
        * int
            Negative log likelihood
        '''
        mixing_coeffs, probs = params
        self.model = (softmax(mixing_coeffs), expit(probs))
        return -self.expected_log_likelihood(batch) / len(batch)

    @jit
    def update(self, i, opt_state, batch):
        '''
        Updates the optimizer state after taking derivative
        i : int
            The current iteration

        opt_state : jax.experimental.optimizers.OptimizerState
            The current state of the parameters

        batch : array
            The subset of observations

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

    def fit_sgd(self, observations, batch_size, rng_key=None, optimizer=None, num_epochs=1):
        '''
        Fits the model using gradient descent algorithm with the given hyperparameters.

        Parameters
        ----------
        observations : array
            The observation sequences which Bernoulli Mixture Model is trained on

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

        * array
            Mixing coefficients found per epoch

        * array
            Probabilities of Bernoulli distribution found per epoch

        * array
            Responsibilites found per epoch
        '''
        global opt_init, opt_update, get_params

        if rng_key is None:
            rng_key = PRNGKey(0)

        if optimizer is not None:
            opt_init, opt_update, get_params = optimizer

        opt_state = opt_init((softmax(self.mixing_coeffs), logit(self.probs)))
        itercount = itertools.count()

        def epoch_step(opt_state, key):

            def train_step(opt_state, batch):
                opt_state, loss = self.update(next(itercount), opt_state, batch)
                return opt_state, loss

            batches = self._make_minibatches(observations, batch_size, key)
            opt_state, losses = scan(train_step, opt_state, batches)

            params = get_params(opt_state)
            mixing_coeffs, probs_logits = params
            probs = expit(probs_logits)
            self.model = (softmax(mixing_coeffs), probs)
            self._probs = probs

            return opt_state, (losses.mean(), *params, self.responsibilities(observations))

        epochs = split(rng_key, num_epochs)
        opt_state, history = scan(epoch_step, opt_state, epochs)
        params = get_params(opt_state)
        mixing_coeffs, probs_logits = params
        probs = expit(probs_logits)
        self.model = (softmax(mixing_coeffs), probs)
        self._probs = probs
        return history

    def plot(self, n_row, n_col, file_name):
        '''
        Plots the mean of each Bernoulli distribution as an image.

        Parameters
        ----------
        n_row : int
            The number of rows of the figure
        n_col : int
            The number of columns of the figure
        file_name : str
            The path where the figure will be stored
        '''
        if n_row * n_col != len(self.mixing_coeffs):
            raise TypeError('The number of rows and columns does not match with the number of component distribution.')
        fig, axes = plt.subplots(n_row, n_col)

        for (coeff, mean), ax in zip(zip(self.mixing_coeffs, self.probs), axes.flatten()):
            ax.imshow(mean.reshape(28, 28), cmap=plt.cm.gray)
            ax.set_title("%1.2f" % coeff)
            ax.axis("off")

        fig.tight_layout(pad=1.0)
        pml.savefig(f"{file_name}.pdf")
        plt.show()