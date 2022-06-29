
import matplotlib.pyplot as plt
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, tree_leaves, tree_map, vmap
from jax.random import split, PRNGKey, permutation
from jax import random

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import tensorflow_datasets as tfds

import flax
import flax.linen as nn

import optax
from sgmcmc_utils import build_optax_optimizer

class BanditEnvironment:
    def __init__(self, key, X, Y):
        # Randomise dataset rows 
        n_obs, n_features = X.shape
        key, mykey = split(key)
        new_ixs = random.choice(mykey, n_obs, (n_obs,), replace=False)
        X = jnp.asarray(X)[new_ixs]
        Y = jnp.asarray(Y)[new_ixs]
        self.contexts = X
        self.labels_onehot = Y


    def get_context(self, t):
        return self.contexts[t]

    def get_reward(self, t, action):
        return np.float(self.labels_onehot[t][action])

    def warmup(self, num_pulls):
        num_steps, num_actions = self.labels_onehot.shape
        # Create array of round-robin actions: 0, 1, 2, 0, 1, 2, 0, 1, 2, ...
        warmup_actions = jnp.arange(num_actions)
        warmup_actions = jnp.repeat(warmup_actions, num_pulls).reshape(num_actions, -1)
        warmup_actions = warmup_actions.reshape(-1, order="F")
        num_warmup_actions, *_ = warmup_actions.shape
        actions = [int(a) for a in warmup_actions]
        contexts = []
        rewards = []
        for t, a in enumerate(actions):
            context = self.get_context(t)
            reward = self.get_reward(t, a)
            contexts.append(context)
            rewards.append(reward)
        return contexts, actions, rewards




class LinearBandit:
    def __init__(self, num_features, num_arms):
        self.num_features = num_features
        self.num_arms = num_arms

    def init_bel(self, key, contexts, actions, rewards):
        eta = 6.0
        lmbda = 0.25
        bel = {
            "mu": jnp.zeros((self.num_arms, self.num_features)),
            "Sigma": 1 * lmbda * jnp.eye(self.num_features) * jnp.ones((self.num_arms, 1, 1)),
            "a": eta * jnp.ones(self.num_arms),
            "b": eta * jnp.ones(self.num_arms),
        }
        nwarmup = len(rewards)
        for t in range(nwarmup): # could do batch update
            context = contexts[t]
            action = actions[t]
            reward = rewards[t]
            bel = self.update_bel(key, bel, context, action, reward)
        return bel

    def update_bel(self, key, bel, context, action, reward):        
        mu_k = bel["mu"][action]
        Sigma_k = bel["Sigma"][action]
        Lambda_k = jnp.linalg.inv(Sigma_k)
        a_k = bel["a"][action]
        b_k = bel["b"][action]
        
        # weight params
        Lambda_update = jnp.outer(context, context) + Lambda_k
        Sigma_update = jnp.linalg.inv(Lambda_update)
        mu_update = Sigma_update @ (Lambda_k @ mu_k + context * reward)
        # noise params
        a_update = a_k + 1/2
        b_update = b_k + (reward ** 2 + mu_k.T @ Lambda_k @ mu_k - mu_update.T @ Lambda_update @ mu_update) / 2
        
        # Update only the chosen action at time t
        mu = jax.ops.index_update(bel["mu"], action, mu_update)
        Sigma = jax.ops.index_update(bel["Sigma"], action, Sigma_update)
        a = jax.ops.index_update(bel["a"], action, a_update)
        b = jax.ops.index_update(bel["b"], action, b_update)
        
        bel = {"mu": mu, "Sigma": Sigma, "a": a, "b": b}
        return bel

    def sample_params(self, key, bel):
        key_sigma, key_w = random.split(key, 2)
        sigma2_samp = tfd.InverseGamma(concentration=bel["a"], scale=bel["b"]).sample(seed=key_sigma)
        cov_matrix_samples = sigma2_samp[:, None, None] * bel["Sigma"]
        w_samp = tfd.MultivariateNormalFullCovariance(loc=bel["mu"], covariance_matrix=cov_matrix_samples).sample(seed=key_w)
        return sigma2_samp, w_samp

    def choose_action(self, key, bel, context):
        # Thompson sampling strategy
        # Could also use epsilon greedy or UCB
        sigma2_samp, w_samp = self.sample_params(key, bel)
        predicted_reward = jnp.einsum("m,km->k", context, w_samp)
        action =  predicted_reward.argmax()
        return action
    

class MLP(nn.Module):
    num_features: int
    num_arms: int
    @nn.compact
    def __call__(self, x): # x has both context and action
        x = nn.relu(nn.Dense(100)(x))
        x = nn.relu(nn.Dense(50)(x))        
        x = nn.Dense(1)(x) # identity activation for scalar regression output
        return x


def fit_model(key, model, X, y, variables):
    opt = optax.adam(learning_rate=1e-1)
    data = (X,y)
    batch_size = 512
    nsteps = 100

    def loglik(params, x, y):
        pred_y = model.apply(variables, x)
        loss = jnp.square(y - pred_y)
        return loss

    def logprior(params):
        # Spherical Gaussian prior
        l2_regularizer = 0.01
        leaves_of_params = tree_leaves(params)
        return sum(tree_map(lambda p: jnp.sum(jax.scipy.stats.norm.logpdf(p, scale=l2_regularizer)), leaves_of_params))

    optimizer = build_optax_optimizer(opt, loglik, logprior, data, batch_size, pbar=False)
    key, mykey = split(key)
    params = variables["params"]
    params, log_post_trace = optimizer(mykey, nsteps, params)
    variables["params"] = params
    return variables


def NeuralGreedy():
    def __init__(self, num_features, num_arms, epsilon, memory=None):
        self.num_features = num_features
        self.num_arms = num_arms
        self.model = MLP(num_features, num_arms)
        self.epsilon = epsilon
        self.memory = memory

    def encode(self, context, action):
        action_onehot = jax.nn.one_hot(action, self.num_arms)
        ndims = self.num_features + self.num_arms
        x = np.concatenate([context, action_onehot]);
        return x

    def init_bel(self, key, contexts, actions, rewards):
        ndims = self.num_features + self.num_arms
        ndata = len(rewards)
        X = jax.vmap(self.encode)(contexts, actions)
        y = rewards
        variables = self.model.init(key, X)
        variables = fit_model(key, self.model, X, y, variables)
        bel = (X, y, variables)
        return bel       

    def update_bel(self, key, bel, context, action, reward): 
        (X, y, variables) = bel
        if self.memory is not None: # finite memory
            if len(y)==self.memory: # memory is full
                X.pop(0)
                y.pop(0)
        x = self.encode(context, action)
        X.append(x)
        y.append(reward)
        variables = fit_model(key, self.model, X, y, variables)
        bel = (X, y, variables)
        return bel

    def choose_action(self, key, bel, context):
        (X, y, variables) = bel
        key, mykey = split(key)
        coin = jax.random.bernoulli(mykey, self.epsilon, (1))
        if coin == 0:
            # random action
            actions = jnp.arange(self.num_arms)
            key, mykey = split(key)
            action = jax.random.choice(mykey, actions)
        else:
            # greedy action
            predicted_rewards = jnp.zeros((self.num_arms,))
            # should make this a minibatch of A examples
            # so we can predict all rewards in parallel
            for a in range(self.num_arms):
                x = self.encode(context, a)
                predicted_rewards[a] = self.model.apply(variables, x)
            action =  predicted_rewards.argmax()
        return action
        
        
    


def run_bandit(key, bandit, env, nsteps, npulls):
    contexts, actions, rewards = env.warmup(npulls)
    nwarmup = len(rewards)
    key, mykey = split(key)
    bel = bandit.init_bel(mykey, contexts, actions, rewards)
    for i in range(nsteps - nwarmup):
        t = nwarmup + i
        print(f'step {t}')
        context = env.get_context(t)
        key, mykey = split(key)
        action = bandit.choose_action(mykey, bel, context)
        reward = env.get_reward(t, action)
        key, mykey = split(key)
        bel = bandit.update_bel(mykey, bel, context, action, reward)
        contexts.append(context)
        actions.append(action)
        rewards.append(reward)
    return contexts, actions, rewards


def get_datasets():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds

def get_mnist():
    train_ds, test_ds = get_datasets()
    train_ds["image"] = train_ds["image"].reshape(-1, 28 ** 2)
    test_ds["image"] = test_ds["image"].reshape(-1, 28 ** 2)

    num_arms = len(jnp.unique(train_ds["label"]))
    num_obs, num_features = train_ds["image"].shape

    train_ds["X"] = train_ds.pop("image")
    train_ds["Y"] = jax.nn.one_hot(train_ds.pop("label"), num_arms)

    test_ds["X"] = test_ds.pop("image")
    test_ds["Y"] = jax.nn.one_hot(test_ds.pop("label"), num_arms)

    num_train = 5000
    X = train_ds["X"][:num_train]
    Y = train_ds["Y"][:num_train]
    return X, Y

X, Y = get_mnist()

# test the code
key = random.PRNGKey(314)
env = BanditEnvironment(key, X, Y)
contexts, actions, rewards = env.warmup(2)
print(len(contexts))
print(contexts[0].shape)

num_obs, num_features = X.shape
_, num_arms = Y.shape
bandit = LinearBandit(num_features, num_arms)

# main loop
contexts, actions, rewards = run_bandit(key, bandit, env, nsteps=20, npulls=1)
print(len(rewards))

# multiple trials
'''
ntrials = 2
keys = random.split(key, ntrials)
npulls = 1
nsteps = 12
def trial(key):    
    env = MyEnvironment(key, X, Y)
    contexts, actions, rewards = run_bandit(key, bandit, env, nsteps, npulls)
    return jnp.array(42)

res = vmap(trial, in_axes=(0,))(keys)
print(res)
'''

# Neural greedy
'''
bandit = NeuralGreedy(num_features, num_arms, epsilon=0.1)
contexts, actions, rewards = run_bandit(key, bandit, env, nsteps=20, npulls=1)
print(len(rewards))
'''
