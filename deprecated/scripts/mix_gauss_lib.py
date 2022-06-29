'''
Implementation of Gaussian Mixture Models.
Author : Aleyna Kara(@karalleyna)
'''

import superimport

import jax.numpy as jnp
from jax import vmap, value_and_grad, jit
from jax.lax import scan
from jax.random import PRNGKey, uniform, split, permutation
from jax.nn import softmax

import distrax
from distrax._src.utils import jittable

import tensorflow_probability as tfp
from mixture_lib import MixtureSameFamily
import matplotlib.pyplot as plt
import itertools
from jax.experimental import optimizers

opt_init, opt_update, get_params = optimizers.adam(5e-2)

class GMM(jittable.Jittable):
    def __init__(self, mixing_coeffs, means, covariances):
        '''
        Initializes Gaussian Mixture Model

        Parameters
        ----------
        mixing_coeffs : array

        means : array

        variances : array
        '''
        self.model = (mixing_coeffs, means, covariances)

    @property
    def mixing_coeffs(self):
        return self._model.mixture_distribution.probs

    @property
    def means(self):
        return self._model.components_distribution.loc

    @property
    def covariances(self):
        return self._model.components_distribution.covariance()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        mixing_coeffs, means, covariances = value
        components_distribution = distrax.as_distribution(
            tfp.substrates.jax.distributions.MultivariateNormalFullCovariance(loc=means,
                                                                              covariance_matrix=covariances,
                                                                              validate_args=True))
        self._model = MixtureSameFamily(mixture_distribution=distrax.Categorical(probs=mixing_coeffs),
                                        components_distribution=components_distribution)

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
        return jnp.sum(self._model.log_prob(observations))

    def responsibility(self, observations, comp_dist_idx):
        '''
         Computes responsibilities, or posterior probability p(z_{comp_dist_idx}|x)

         Parameters
         ----------
         observations : array(N, seq_len)
             Dataset

         comp_dist_idx : int
            Index which specifies the specific mixing distribution component

         Returns
         -------
         * array
             Responsibilities
         '''
        return self._model.posterior_marginal(observations).prob(comp_dist_idx)

    def responsibilities(self, observations):
        '''
         Computes responsibilities, or posterior probability p(z|x)

         Parameters
         ----------
         observations : array(N, seq_len)
             Dataset

         Returns
         -------
         * array
             Responsibilities
         '''
        return self.model.posterior_marginal(observations).probs

    def _m_step(self, observations, S, eta):
        '''
        Maximization step

        Parameters
        ----------
        observations : array(N, seq_len)
            Dataset

        S : array
            A prior p(theta) is defined over the parameters to find MAP solutions

        eta : int

        Returns
        -------
        * array
            Mixing coefficients

        * array
            Means

        * array
            Covariances
        '''
        n_obs, n_comp = observations.shape

        def m_step_per_gaussian(responsibility):
            effective_prob = responsibility.sum()
            mean = (responsibility[:, None] * observations).sum(axis=0) / effective_prob

            centralized_observations = (observations - mean)
            covariance = responsibility[:, None, None] * jnp.einsum("ij, ik->ikj",
                                                                    centralized_observations,
                                                                    centralized_observations)

            covariance = covariance.sum(axis=0)

            if eta is None:
                covariance = covariance / effective_prob
            else:
                covariance = (S + covariance) / (eta + effective_prob + n_comp + 2)

            mixing_coeff = effective_prob / n_obs
            return (mixing_coeff, mean, covariance)

        mixing_coeffs, means, covariances = vmap(m_step_per_gaussian, in_axes=(1))(self.responsibilities(observations))
        return mixing_coeffs, means, covariances

    def _add_final_values_to_history(self, history, observations):
        '''
        Appends the final values of log likelihood, mixing coefficients, means, variances and responsibilities into the
        history

        Parameters
        ----------
        history : tuple
            Consists of values of log likelihood, mixing coefficients, means, variances and responsibilities, which are
            found per iteration

        observations : array(N, seq_len)
            Dataset

        Returns
        -------
        * array
            Mean loss values found per iteration

        * array
            Mixing coefficients found per iteration

        * array
            Means of Gaussian distribution found per iteration

        * array
            Covariances of Gaussian distribution found per iteration

        * array
            Responsibilites found per iteration
        '''
        ll_hist, mix_dist_probs_hist, comp_dist_loc_hist, comp_dist_cov_hist, responsibility_hist = history

        ll_hist = jnp.append(ll_hist, self.expected_log_likelihood(observations))
        mix_dist_probs_hist = jnp.vstack([mix_dist_probs_hist, self.mixing_coeffs])
        comp_dist_loc_hist = jnp.vstack([comp_dist_loc_hist, self.means[None, :]])
        comp_dist_cov_hist = jnp.vstack([comp_dist_cov_hist, self.covariances[None, :]])
        responsibility_hist = jnp.vstack([responsibility_hist, jnp.array([self.responsibility(observations, 0)])])

        history = (ll_hist, mix_dist_probs_hist, comp_dist_loc_hist, comp_dist_cov_hist, responsibility_hist)

        return history


    def fit_em(self, observations, num_of_iters, S=None, eta=None):
        '''
        Fits the model using em algorithm.

        Parameters
        ----------
        observations : array(N, seq_len)
            Dataset

        num_of_iters : int
            The number of iterations the training process takes place

        S : array
            A prior p(theta) is defined over the parameters to find MAP solutions

        eta : int

        Returns
        -------
        * array
            Mean loss values found per iteration

        * array
            Mixing coefficients found per iteration

        * array
            Means of Gaussian distribution found per iteration

        * array
            Covariances of Gaussian distribution found per iteration

        * array
            Responsibilites found per iteration
        '''
        initial_mixing_coeffs = self.mixing_coeffs
        initial_means = self.means
        initial_covariances = self.covariances

        iterations = jnp.arange(num_of_iters)

        def train_step(params, i):
            self.model = params
            log_likelihood = self.expected_log_likelihood(observations)
            responsibility = self.responsibility(observations, 0)

            mixing_coeffs, means, covariances = self._m_step(observations, S, eta)
            return (mixing_coeffs, means, covariances), (log_likelihood, *params, responsibility)

        initial_params = (initial_mixing_coeffs,
                          initial_means,
                          initial_covariances)

        final_params, history = scan(train_step, initial_params, iterations)
        self.model = final_params

        history = self._add_final_values_to_history(history, observations)

        return history

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

    def _transform_to_covariance_matrix(self, sq_mat):
        '''
        Takes the upper triangular matrix of the given matrix and then multiplies it by its transpose
        https://ericmjl.github.io/notes/stats-ml/estimating-a-multivariate-gaussians-parameters-by-gradient-descent/

        Parameters
        ----------
        sq_mat : array
            Square matrix

        Returns
        -------
        * array
        '''
        U = jnp.triu(sq_mat)
        U_T = jnp.transpose(U)
        return jnp.dot(U_T, U)

    def loss_fn(self, params, batch):
        """
        Calculates expected mean negative loglikelihood.

        Parameters
        ----------
        params : tuple
            Consists of mixing coefficients' logits, means and variances of the Gaussian distributions respectively.

        batch : array
            The subset of observations

        Returns
        -------
        * int
            Negative log likelihood
        """
        mixing_coeffs, means, untransormed_cov = params
        cov_matrix = vmap(self._transform_to_covariance_matrix)(untransormed_cov)
        self.model = (softmax(mixing_coeffs), means, cov_matrix)
        return -self.expected_log_likelihood(batch) / len(batch)

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

    def fit_sgd(self, observations, batch_size, rng_key=None, optimizer=None, num_epochs=3):
        '''
        Finds the parameters of Gaussian Mixture Model using gradient descent algorithm with the given hyperparameters.

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
            Means of Gaussian distribution found per epoch

        * array
            Covariances of Gaussian distribution found per epoch

        * array
            Responsibilites found per epoch
        '''
        global opt_init, opt_update, get_params

        if rng_key is None:
            rng_key = PRNGKey(0)

        if optimizer is not None:
            opt_init, opt_update, get_params = optimizer

        opt_state = opt_init((softmax(self.mixing_coeffs), self.means, self.covariances))
        itercount = itertools.count()

        def epoch_step(opt_state, key):

            def train_step(opt_state, batch):
                opt_state, loss = self.update(next(itercount), opt_state, batch)
                return opt_state, loss

            batches = self._make_minibatches(observations, batch_size, key)
            opt_state, losses = scan(train_step, opt_state, batches)

            params = get_params(opt_state)
            mixing_coeffs, means, untransormed_cov = params
            cov_matrix = vmap(self._transform_to_covariance_matrix)(untransormed_cov)
            self.model = (softmax(mixing_coeffs), means, cov_matrix)
            responsibilities = self.responsibilities(observations)

            return opt_state, (losses.mean(), *params, responsibilities)

        epochs = split(rng_key, num_epochs)
        opt_state, history = scan(epoch_step, opt_state, epochs)

        params = get_params(opt_state)
        mixing_coeffs, means, untransormed_cov = params
        cov_matrix = vmap(self._transform_to_covariance_matrix)(untransormed_cov)
        self.model = (softmax(mixing_coeffs), means, cov_matrix)

        return history


    def plot(self, observations, means=None, covariances=None, responsibilities=None,
                      step=0.01, cmap="viridis", colors=None, ax=None):
        '''
        Plots Gaussian Mixture Model.

        Parameters
        ----------
        observations : array
            Dataset

        means : array

        covariances : array

        responsibilities : array

        step: float
            Step size of the grid for the density contour.

        cmap : str

        ax : array
        '''
        means = self.means if means is None else means
        covariances = self.covariances if covariances is None else covariances
        responsibilities = self.model.posterior_marginal(observations).probs if responsibilities is None \
            else responsibilities

        colors = uniform(PRNGKey(100), (means.shape[0], 3)) if colors is None else colors
        ax = ax if ax is not None else plt.subplots()[1]

        min_x, min_y = observations.min(axis=0)
        max_x, max_y = observations.max(axis=0)

        xs, ys = jnp.meshgrid(jnp.arange(min_x, max_x, step), jnp.arange(min_y, max_y, step))
        grid = jnp.vstack([xs.ravel(), ys.ravel()]).T

        def multivariate_normal(mean, cov):
            '''
            Initializes multivariate normal distribution with the given mean and covariance.
            Note that the pdf has the same precision with its parameters' dtype.
            '''
            return tfp.substrates.jax.distributions.MultivariateNormalFullCovariance(loc=mean,
                                                                                     covariance_matrix=cov)

        for (means, cov), color in zip(zip(means, covariances), colors):
            normal_dist = multivariate_normal(means, cov)
            density = normal_dist.prob(grid).reshape(xs.shape)
            ax.contour(xs, ys, density, levels=1, colors=color, linewidths=5)

        ax.scatter(*observations.T, alpha=0.7, c=responsibilities, cmap=cmap, s=10)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)