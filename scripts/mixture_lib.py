# Mixture distributions
# Author Aleyna Kara(@karalleyna)

import superimport
#!pip install distrax
import distrax

import jax
import jax.numpy as jnp

class MixtureSameFamily(distrax.MixtureSameFamily):
    def _per_mixture_component_log_prob(self, value):
        '''Per mixture component log probability.
        https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/mixture_same_family.py

        Parameters
        ----------
          value: array
            Represents observations from the mixture. Must
            be broadcastable with the mixture's batch shape.

        Returns
        -------
          *array
          Represents, for each observation and for each mixture
          component, the log joint probability of that mixture component and
          the observation. The shape will be equal to the concatenation of (1) the
          broadcast shape of the observations and the batch shape, and (2) the
          number of mixture components.
        '''
        # Add component axis to make input broadcast with components distribution.
        expanded = jnp.expand_dims(value, axis=-1 - len(self.event_shape))
        # Compute `log_prob` in every component.
        lp = self.components_distribution.log_prob(expanded)
        # Last batch axis is number of components, i.e. last axis of `lp` below.
        # Last axis of mixture log probs are components.
        return lp + self._mixture_log_probs

    def log_prob(self, value):
        '''See "distrax.Distribution.log_prob and distrax.MixtureSameFamily".
        https://github.com/deepmind/distrax/blob/master/distrax/_src/distributions/distribution.py
        https://github.com/deepmind/distrax/blob/master/distrax/_src/distributions/mixture_same_family.py
        '''
        # Reduce last axis of mixture log probs are components
        return jax.scipy.special.logsumexp(self._per_mixture_component_log_prob(value), axis=-1)

    def posterior_marginal(self, observations):
        '''Compute the marginal posterior distribution for a batch of observations.
        https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MixtureSameFamily?version=nightly#posterior_marginal

        Parameters
        ----------
          observations:
            An array representing observations from the mixture. Must
            be broadcastable with the mixture's batch shape.

        Returns
        -------
          * array
            Posterior marginals that is a `Categorical` distribution object representing
            the marginal probability of the components of the mixture. The batch
            shape of the `Categorical` will be the broadcast shape of `observations`
            and the mixture batch shape; the number of classes will equal the
            number of mixture components.
        '''
        return distrax.Categorical(logits=self._per_mixture_component_log_prob(observations))

    def posterior_mode(self, observations):
        '''Compute the posterior mode for a batch of distributions.
        https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MixtureSameFamily?version=nightly#posterior_mode

        Parameters
        ----------
          observations:
            Represents observations from the mixture. Must
            be broadcastable with the mixture's batch shape.

        Returns
        -------
          * array
            Represents the mode (most likely component) for each
            observation. The shape will be equal to the broadcast shape of the
            observations and the batch shape.
        '''
        return jnp.argmax(self._per_mixture_component_log_prob(observations), axis=-1)