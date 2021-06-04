# Occasionally dishonest casino example [Durbin98, p54]. This script
# exemplifies a Hidden Markov Model (HMM) in which the throw of a die
# may result in the die being biased (towards 6) or unbiased. If the dice turns out to
# be biased, the probability of remaining biased is high, and similarly for the unbiased state.
# Assuming we observe the die being thrown n times the goal is to recover the periods in which
# the die was biased.
# Original matlab code: https://github.com/probml/pmtk3/blob/master/demos/casinoDemo.m
# Author: Gerardo Duran-Martin (@gerdm)

import hmm_lib as hmm
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed, choice
import pyprobml_utils as pml

A = np.array([
    [0.95, 0.05],
    [0.10, 0.90]
])

px = np.array([
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], # fair die
    [1/10, 1/10, 1/10, 1/10, 1/10, 5/10] # loaded die
])

n_samples = 300
π = np.array([1, 1]) / 2
casino = hmm.HMMDiscrete(A, px, π)
z_hist, x_hist = casino.sample(n_samples, 314)

z_hist_str = "".join((z_hist + 1).astype(str))[:60]
x_hist_str = "".join((x_hist + 1).astype(str))[:60]

print("Printing sample observed/latent...")
print(f"x: {x_hist_str}")
print(f"z: {z_hist_str}")

res = casino.filter_smooth(x_hist)
z_map = casino.map_state(x_hist)
filtering = res["filtering"]
smoothing = res["smoothing"]

fig, ax = plt.subplots()
casino.plot_inference(filtering, z_hist, ax)
ax.set_ylabel("p(loaded)")
ax.set_title("Filtered")
pml.savefig("hmm_casino_filter.pdf")

fig, ax = plt.subplots()
casino.plot_inference(smoothing, z_hist, ax)
ax.set_ylabel("p(loaded)")
ax.set_title("Smoothed")
pml.savefig("hmm_casino_smooth.pdf")

fig, ax = plt.subplots()
casino.plot_inference(z_map, z_hist, ax, map_estimate=True)
ax.set_ylabel("MAP state")
ax.set_title("Viterbi")
pml.savefig("hmm_casino_map.pdf")

plt.show()
