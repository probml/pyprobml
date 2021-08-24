# Blind source separation using FastICA and PCA
# Author : Aleyna Kara
# This file is based on https://github.com/probml/pmtk3/blob/master/demos/icaDemo.m

import superimport

from sklearn.decomposition import PCA, FastICA
import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml

def plot_signals(signals, suptitle, file_name):
  plt.figure(figsize=(8, 4))
  for i, signal in enumerate(signals, 1):
    plt.subplot(n_signals, 1, i)
    plt.plot(signal)
    plt.xlim([0, N])
    plt.tight_layout()
  plt.suptitle(suptitle)
  plt.subplots_adjust(top=0.85)
  pml.savefig(f'{file_name}.pdf')
  plt.show()

# https://github.com/davidkun/FastICA/blob/master/demosig.m
def generate_signals():
  v = np.arange(0, 500)
  signals = np.zeros((n_signals, N))

  signals[0, :] = np.sin(v/2) # sinusoid
  signals[1, :] = ((v % 23 - 11) / 9)**5
  signals[2, :] = ((v % 27 - 13)/ 9) # sawtooth

  rand = np.random.rand(1, N)
  signals[3, :] = np.where(rand < 0.5, rand * 2 -1, -1) * np.log(np.random.rand(1, N)) #impulsive noise

  signals /= signals.std(axis=1).reshape((-1,1))
  signals -= signals.mean(axis=1).reshape((-1,1))
  A = np.random.rand(n_signals, n_signals) # mixing matrix
  return signals, A @ signals

np.random.seed(0)
n_signals, N = 4, 500
signals, mixed_signals = generate_signals()

plot_signals(signals, 'Truth', 'ica-truth')

plot_signals(mixed_signals, 'Observed Signals', 'ica-obs')

pca = PCA(whiten=True, n_components=4)
signals_pca = pca.fit(mixed_signals.T).transform(mixed_signals.T)

ica = FastICA(algorithm='deflation', n_components=4)
signals_ica = ica.fit_transform(mixed_signals.T)

plot_signals(signals_pca.T, 'PCA estimate','ica-pca')

plot_signals(signals_ica.T, 'ICA estimate', 'ica-ica')