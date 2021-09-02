# Occasionally dishonest casino example [Durbin98, p54]. This script
# exemplifies a Hidden Markov Model (HMM) in which the throw of a die
# may result in the die being biased (towards 6) or unbiased. If the dice turns out to
# be biased, the probability of remaining biased is high, and similarly for the unbiased state.
# Assuming we observe the die being thrown n times the goal is to recover the periods in which
# the die was biased.
# Original matlab code: https://github.com/probml/pmtk3/blob/master/demos/casinoDemo.m
# Author: Gerardo Duran-Martin (@gerdm)

import superimport

from hmm_discrete_lib import HMMNumpy
from hmm_discrete_lib import hmm_sample_numpy, hmm_forwards_backwards_numpy, hmm_viterbi_numpy

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from hmm_discrete_lib import hmm_plot_graphviz

def find_dishonest_intervals(z_hist):
    """
    Find the span of timesteps that the
    simulated systems turns to be in state 1
    Parameters
    ----------
    z_hist: array(n_samples)
        Result of running the system with two
        latent states
    Returns
    -------
    list of tuples with span of values
    """
    spans = []
    x_init = 0
    for t, _ in enumerate(z_hist[:-1]):
        if z_hist[t + 1] == 0 and z_hist[t] == 1:
            x_end = t
            spans.append((x_init, x_end))
        elif z_hist[t + 1] == 1 and z_hist[t] == 0:
            x_init = t + 1
    return spans

def plot_inference(inference_values, z_hist, ax, state=1, map_estimate=False):
    """
    Plot the estimated smoothing/filtering/map of a sequence of hidden states.
    "Vertical gray bars denote times when the hidden
    state corresponded to state 1. Blue lines represent the
    posterior probability of being in that state given diﬀerent subsets
    of observed data." See Markov and Hidden Markov models section for more info
    Parameters
    ----------
    inference_values: array(n_samples, state_size)
        Result of runnig smoothing method
    z_hist: array(n_samples)
        Latent simulation
    ax: matplotlib.axes
    state: int
        Decide which state to highlight
    map_estimate: bool
        Whether to plot steps (simple plot if False)
    """
    n_samples = len(inference_values)
    xspan = np.arange(1, n_samples + 1)
    spans = find_dishonest_intervals(z_hist)
    if map_estimate:
        ax.step(xspan, inference_values, where="post")
    else:
        ax.plot(xspan, inference_values[:, state])

    for span in spans:
        ax.axvspan(*span, alpha=0.5, facecolor="tab:gray", edgecolor="none")
    ax.set_xlim(1, n_samples)
    #ax.set_ylim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Observation number")

# state transition matrix
A = np.array([
    [0.95, 0.05],
    [0.10, 0.90]
])

# observation matrix
B = np.array([
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], # fair die
    [1/10, 1/10, 1/10, 1/10, 1/10, 5/10] # loaded die
])

n_samples = 300
init_state_dist = np.array([1, 1]) / 2
params = HMMNumpy(A, B, init_state_dist)
z_hist, x_hist = hmm_sample_numpy(params, n_samples, 314)

z_hist_str = "".join((z_hist + 1).astype(str))[:60]
x_hist_str = "".join((x_hist + 1).astype(str))[:60]

print("Printing sample observed/latent...")
print(f"x: {x_hist_str}")
print(f"z: {z_hist_str}")

# Do inference
alpha, _, gamma, loglik = hmm_forwards_backwards_numpy(params, x_hist, len(x_hist))
print(f'Loglikelihood: {loglik}')

z_map = hmm_viterbi_numpy(params, x_hist)

# Plot results
fig, ax = plt.subplots()
plot_inference(alpha, z_hist, ax)
ax.set_ylabel("p(loaded)")
ax.set_title("Filtered")
pml.savefig("hmm_casino_filter.pdf")

fig, ax = plt.subplots()
plot_inference(gamma, z_hist, ax)
ax.set_ylabel("p(loaded)")
ax.set_title("Smoothed")
pml.savefig("hmm_casino_smooth.pdf")

fig, ax = plt.subplots()
plot_inference(z_map, z_hist, ax, map_estimate=True)
ax.set_ylabel("MAP state")
ax.set_title("Viterbi")
pml.savefig("hmm_casino_map.pdf")
plt.show()

file_name = 'hmm_casino_params'
states, observations = ['Fair Dice', 'Loaded Dice'], [str(i+1) for i in range(B.shape[1])]
hmm_plot_graphviz(params, file_name, states, observations)