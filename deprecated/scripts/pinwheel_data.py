# Mostly copied from:
# https://github.com/mattjj/svae/blob/master/experiments/gmm_svae_synth.py

import superimport

import numpy as np
import matplotlib.pyplot as plt

def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate,     rand = np.random.RandomState(0)):
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    features = rand.randn(num_classes * num_per_class, 2) \
               * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles),
                          np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    data = 10 * rand.permutation(np.einsum('ti,tij->tj', features, rotations))
    return data.astype(np.float32)


data = make_pinwheel_data(
    radial_std=0.3,
    tangential_std=0.05,
    num_classes=5,
    num_per_class=100,
    rate=0.25)

plt.plot(data[:, 0], data[:, 1], 'k.', markersize=3);