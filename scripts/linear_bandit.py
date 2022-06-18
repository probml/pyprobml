import enum

import jax.numpy as jnp
from jax import lax
from jax import random

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class ExplorationPolicy(enum.Enum):
    """Possible exploration policies."""

    linear_ucb_policy = 1
    linear_thompson_sampling_policy = 2
    linear_epsilon_greedy_policy = 3


class LinearBandit:
    def __init__(self, num_features, num_arms, exploration_policy,
                 eta=6.0, lmbda=0.25, alpha=1.0, epsilon=0):
        self.num_features = num_features
        self.num_arms = num_arms
        self.eta = eta
        self.lmbda = lmbda
        self._alpha = alpha
        self._epsilon = epsilon

        if exploration_policy == ExplorationPolicy.linear_ucb_policy:
            self._policy = self.ucb_policy
        elif exploration_policy == ExplorationPolicy.linear_thompson_sampling_policy:
            self._policy = self.thompson_sampling_policy
        elif exploration_policy == ExplorationPolicy.linear_epsilon_greedy_policy:
            self._policy = self.epsilon_greedy_policy
        else:
            raise NotImplemented

    def _get_epsilon(self):
        if callable(self._epsilon):
            return self._epsilon()
        else:
            return self._epsilon

    def init_bel(self, contexts, actions, rewards):
        mu = jnp.zeros((self.num_arms, self.num_features))
        Sigma = jnp.eye(self.num_features) * \
            jnp.ones((self.num_arms, 1, 1)) / self.lmbda
        a = self.eta * jnp.ones((self.num_arms,))
        b = self.eta * jnp.ones((self.num_arms,))

        bel = (mu, Sigma, a, b)

        def update(bel, cur):  # could do batch update
            context, action, reward = cur
            bel = self.update_bel(bel, context, action, reward)
            return bel, None

        if contexts and actions and rewards:
            assert len(contexts) == len(actions) == len(rewards)
            bel, _ = lax.scan(update, bel, (contexts, actions, rewards))

        return bel

    def update_bel(self, bel, context, action, reward):
        mu, Sigma, a, b = bel

        mu_k, Sigma_k = mu[action], Sigma[action]
        Lambda_k = jnp.linalg.inv(Sigma_k)
        a_k, b_k = a[action], b[action]

        # weight params
        Lambda_update = jnp.outer(context, context) + Lambda_k
        Sigma_update = jnp.linalg.inv(Lambda_update)
        mu_update = Sigma_update @ (Lambda_k @ mu_k + context * reward)
        # noise params
        a_update = a_k + 1 / 2
        b_update = b_k + (reward ** 2 + mu_k.T @ Lambda_k @
                          mu_k - mu_update.T @ Lambda_update @ mu_update) / 2

        # Update only the chosen action at time t
        mu = mu.at[action].set(mu_update)
        Sigma = Sigma.at[action].set(Sigma_update)
        a = a.at[action].set(a_update)
        b = b.at[action].set(b_update)

        bel = (mu, Sigma, a, b)

        return bel

    def _sample_params(self, key, bel):
        mu, Sigma, a, b = bel

        sigma_key, w_key = random.split(key, 2)
        sigma2_samp = tfd.InverseGamma(
            concentration=a, scale=b).sample(seed=sigma_key)
        covariance_matrix = sigma2_samp[:, None, None] * Sigma
        w = tfd.MultivariateNormalFullCovariance(
            loc=mu, covariance_matrix=covariance_matrix).sample(seed=w_key)
        return w

    def thompson_sampling_policy(self, key, bel, context):
        w = self._sample_params(key, bel)
        predicted_reward = jnp.einsum("m,km->k", context, w)
        action = predicted_reward.argmax()
        return action

    def ucb_policy(self, key, bel, context):
        mu, Sigma, a, b = bel
        covariance_matrix = Sigma * Sigma
        predicted_reward = jnp.einsum("m,km->k", context, mu)
        predicted_variance = jnp.einsum(
            "n,knm,m->k", context, covariance_matrix, context)

        rewards_for_argmax = predicted_reward + \
            self._alpha * jnp.sqrt(predicted_variance)

        return rewards_for_argmax.argmax()

    def greedy_policy(self, key, bel, context):
        mu, Sigma, a, b = bel
        predicted_reward = jnp.einsum("m,km->k", context, mu)

        return predicted_reward.argmax()

    def epsilon_greedy_policy(self, key, bel, context):
        rng = random.uniform(key)
        if rng < self._get_epsilon():
            return random.uniform(key, maxval=self.num_arms)
        else:
            return self.greedy_policy(key, bel, context)

    def choose_action(self, key, bel, context):
        return self._policy(key, bel, context)
