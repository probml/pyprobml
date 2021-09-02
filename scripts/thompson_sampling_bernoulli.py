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


class BetaBernoulliBandit:
    def __init__(self, alpha0, beta0, num_bandits):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.num_bandits = num_bandits
    
    def sample(self, key, alphas, betas):
        actions = random.beta(key, alphas, betas)
        return actions
    
    def predict_reward(self, actions):
        rewards = actions.argmax()

def thompson_sampling_step(state, key):
    alphas, betas, reward_per_arm = state
    K = len(alphas)
    key_action, key_reward = random.split(key)
    
    # Choose an arm to pull
    # (Sample from the policy distribution)
    action_t = random.beta(key_action, alphas, betas).argmax()
    # Pull the arm and observe reward (either 1 or 0)
    reward = random.bernoulli(key_reward, reward_per_arm[action_t])
    
    # Update policy distribution
    ind_vector = one_hot(action_t, K)
    alphas_posterior = alphas + ind_vector * reward
    betas_posterior = betas + ind_vector * (1 - reward)
    
    return (alphas_posterior, betas_posterior, reward_per_arm), (alphas_posterior, betas_posterior)


p_range = jnp.linspace(0, 1, 100)
colors = ["orange", "blue", "green", "red"]
colors = [f"tab:{color}" for color in colors]

T = 200
key = random.PRNGKey(314)
keys = random.split(key, T)
reward_per_arm = jnp.array([0.65, 0.4, 0.5, 0.9])
K = len(reward_per_arm)

alpha_priors = jnp.ones(K) * 1
beta_priors = jnp.ones(K) * 1

init_state = (alpha_priors, beta_priors, reward_per_arm)
posteriors, hist = jax.lax.scan(thompson_sampling_step, init_state, keys)
alpha_posterior, beta_posterior, _ = posteriors
alpha_hist, beta_hist = hist

bandits_pdf_hist = beta.pdf(p_range[:, None, None], alpha_hist[None, ...], beta_hist[None, ...])

# Indexed by position
times = [0, 9, 19, 49, 99, 199]
for t in times:
    for k, color in enumerate(colors):
        fig, axi = plt.subplots()
        bandit = bandits_pdf_hist[:, t, k]
        axi.plot(p_range, bandit, c=color)
        axi.set_xlim(0, 1)
        n_pos = alpha_hist[t, k].item() - 1
        n_trials = beta_hist[t, k].item() + n_pos - 1
        axi.set_title(f"t={t+1}\np={reward_per_arm[k]:0.2f}\n{n_pos:.0f}/{n_trials:.0f}")
        pml.savefig(f"thompson_sampling_bernoulli_w{k}_t{t}.pdf")
        plt.tight_layout()
plt.show()
