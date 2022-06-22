# Resolution of a Multi-Armed Bandit problem
# using Thompson Sampling.
# Author: Gerardo Durán-Martín (@gerdm)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from jax import random
from jax.nn import one_hot
from jax.scipy.stats import beta
from functools import partial


class BetaBernoulliBandits:
    def __init__(self, K):
        self.K = K
        
    def sample(self, key, params):
        alphas = params["alpha"]
        betas = params["beta"]
        params_sample = random.beta(key, alphas, betas)
        return params_sample
    
    def predict_rewards(self, params_sample):
        return params_sample
    
    def update(self, action, params, reward):
        alphas = params["alpha"]
        betas = params["beta"]
        # Update policy distribution
        ind_vector = one_hot(action, self.K)
        alphas_posterior = alphas + ind_vector * reward
        betas_posterior = betas + ind_vector * (1 - reward)
        return {
            "alpha": alphas_posterior,
            "beta": betas_posterior
        }


def true_reward(key, action, mean_rewards):
    reward = random.bernoulli(key, mean_rewards[action])
    return reward


def thompson_sampling_step(model_params, key, model, environment):
    """
    Context-free implementation of the Thompson sampling algorithm.
    This implementation considers a single step
    
    Parameters
    ----------
    model_params: dict
    environment: function
    key: jax.random.PRNGKey
    moidel: instance of a Bandit model
    """
    key_sample, key_reward = random.split(key)
    params = model.sample(key_sample, model_params)
    pred_rewards = model.predict_rewards(params)
    action = pred_rewards.argmax()
    reward = environment(key_reward, action)
    model_params = model.update(action, model_params, reward)
    prob_arm = model_params["alpha"] / (model_params["alpha"] + model_params["beta"])
    return model_params, (model_params, prob_arm)


if __name__ == "__main__":
    T = 200
    key = random.PRNGKey(31415)
    keys = random.split(key, T)
    mean_rewards = jnp.array([0.4, 0.5, 0.2, 0.9])
    K = len(mean_rewards)
    bbbandit = BetaBernoulliBandits(mean_rewards)
    init_params = {"alpha": jnp.ones(K),
                "beta": jnp.ones(K)}

    environment = partial(true_reward, mean_rewards=mean_rewards)
    thompson_partial = partial(thompson_sampling_step,
                            model=BetaBernoulliBandits(K),
                            environment=environment)
    posteriors, (hist, prob_arm_hist) = jax.lax.scan(thompson_partial, init_params, keys)

    plt.plot(prob_arm_hist)
    plt.legend([f"mean reward: {reward:0.2f}" for reward in mean_rewards], loc="lower right")
    pml.savefig("beta-bernoulli-thompson-sampling.pdf")
    plt.show()
