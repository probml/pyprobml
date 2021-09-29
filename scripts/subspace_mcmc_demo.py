'''
This demo shows the subspace learning procedure using various HMC algorithms and then compares them
using agreement and total variation distance metrics.
Author : Aleyna Kara(@karalleyna)
'''

import superimport

import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

import jax
from jax.random import split
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
import jax.numpy as jnp

from subspace_learning_lib import subspace_learning, loglikelihood, logprior, accuracy
from subspace_learning_lib import build_sgd, build_nuts_sampler
import pyprobml_utils as pml

from sgmcmcjax.samplers import *

def load_dataset(filename):
    with open(filename, "rb") as f:
        statlog = pickle.load(f)
    dataset, opt_rewards, opt_actions, num_actions, context_dim, vocab_processor = statlog

    train_ds, test_ds = train_test_split(dataset, test_size=0.2, random_state=314)

    train_ds = {
        "X": jnp.array(train_ds[:, :-num_actions]),
        "y": jnp.array(train_ds[:, -num_actions:])
    }

    test_ds = {
        "X": jnp.array(test_ds[:, :-num_actions]),
        "y": jnp.array(test_ds[:, -num_actions:])

    }

    return train_ds, test_ds, num_actions

# https://github.com/izmailovpavel/neurips_bdl_starter_kit/blob/main/metrics.py
def agreement(predictions, reference):
    """Returns 1 if predictions match and 0 otherwise."""
    return (predictions.argmax(axis=-1) == reference.argmax(axis=-1)).mean()

# https://github.com/izmailovpavel/neurips_bdl_starter_kit/blob/main/metrics.py
def total_variation_distance(predictions, reference):
    """Returns total variation distance."""
    return jnp.abs(predictions - reference).sum(axis=-1).mean() / 2.


def plot_metrics(all_probs, metric, xlabel, min_x=0, filename="metrics"):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ticks = []
    cmap = plt.cm.coolwarm
    nuts_probs = all_probs["NUTS"]

    i = 0
    for k, probs in all_probs.items():
        if k != "NUTS":
            metric_result = metric(nuts_probs, probs)
            ax.barh(i, metric_result, height=0.8, color=cmap((metric_result - min_x) * 10))
            ticks.append(f"NUTS - {k}")
            i += 1

    ax.set_yticks(range(i + 1))
    ax.set_yticklabels(ticks)
    ax.set_xlabel(xlabel, fontsize=12)
    pml.save_fig(f'{filename}.pdf')
    plt.show()

pickle_name = './statlog-bandit-data.pkl'
train_ds, test_ds, num_actions = load_dataset(pickle_name)

train_ds["y"] = jnp.argmax(train_ds["y"], axis=-1)
test_ds["y"] = jnp.argmax(test_ds["y"], axis=-1)

num_train, *num_features = train_ds["X"].shape
num_features = np.product(num_features)

init_random_params, predict = stax.serial(
    Dense(num_features),
    Relu,
    Dense(50),
    Relu,
    Dense(num_actions),
    LogSoftmax
)

key = jax.random.PRNGKey(42)
init_key, key = split(key)
_, params_full_init = init_random_params(init_key, input_shape=(-1, num_features))

d = 10
num_warmup = 1000
num_samples = 3000
batch_size = 1024
optimizer_params = {"lr": 1e-1,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-7}

models = {"SGD": {"sampler_fn": build_sgd,
                  "hyperparams": {"batch_size": batch_size,
                                  "optimizer_params": optimizer_params
                                  }
                  },
          "NUTS": {"sampler_fn": build_nuts_sampler,
                   "hyperparams": {}
                   },

          "SGLD": {"sampler_fn": build_sgld_sampler,
                   "hyperparams": {"batch_size": batch_size,
                                   "dt": 1e-6}
                   },

          "SGLDCV": {"sampler_fn": build_sgldCV_sampler,
                     "hyperparams": {"batch_size": batch_size,
                                     "dt": 1e-8,
                                     "centering_value": None},
                     "optimizer_params": optimizer_params
                     },

          "SGHMC": {"sampler_fn": build_sghmc_sampler,
                    "hyperparams": {"batch_size": batch_size,
                                    "dt": 1e-6,
                                    "L": 5}
                    },

          "SGHMCCV": {"sampler_fn": build_sghmcCV_sampler,
                      "hyperparams": {"batch_size": batch_size,
                                      "dt": 1e-8,
                                      "L": 5,
                                      "centering_value": None},
                      "optimizer_params": optimizer_params
                      },
          "SGNHT": {"sampler_fn": build_sgnht_sampler,
                    "hyperparams": {"batch_size": batch_size,
                                    "dt": 1e-6,
                                    "a": 0.02}
                    },
          "SGNHTCV": {"sampler_fn": build_sgnhtCV_sampler,
                      "hyperparams": {"batch_size": batch_size,
                                      "dt": 1e-7,
                                      "a": 0.02,
                                      "centering_value": None},
                      "optimizer_params": optimizer_params
                      },
          "ULA": {"sampler_fn": build_sgld_sampler,
                  "hyperparams": {"batch_size": num_train,
                                  "dt": 1e-4
                                  }
                  }
          }

all_probs = {}

for model_name, params in models.items():
    run_key, key = split(key)
    print(model_name)
    final_full_params = subspace_learning(run_key, num_samples, num_warmup, train_ds, d, \
                                          loglikelihood, logprior, params_full_init, \
                                          params["sampler_fn"], predict, params["hyperparams"])
    training_acc, probs = accuracy(final_full_params, (train_ds["X"], train_ds["y"]), predict_fn=predict)
    all_probs[f"{model_name}"] = probs
    print("Training Accuracy : {:.2%}".format(training_acc))

plot_metrics(all_probs, agreement, "Agreement", min_x=0.7, filename="shuttle_mcmc_agreement")
plot_metrics(all_probs, total_variation_distance, "Total Variation Distance", min_x=0.7, filename="shuttle_mcmc_var_dist")