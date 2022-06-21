# Tests min and variance to check whether Newcomb's speed of light data is Gaussian or not
# Author : Aleyna Kara
# This file is generated from https://github.com/probml/pmtk3/blob/master/demos/newcombPlugin.m

import superimport

import pyprobml_utils as pml
import numpy as np
import requests
import matplotlib.pyplot as plt

def plot_posterior(test_val, test_val_true, title, file_name):
  plt.hist(test_val)
  plt.axvline(x=test_val_true, c='r')
  plt.title(title)
  pml.savefig(f'{file_name}.pdf')
  plt.show()

# read data
url= "http://www.stat.columbia.edu/~gelman/book/data/light.asc"
s = requests.get(url).content.decode('utf-8').strip().split('\n')[-7:]
D = np.array(' '.join(s).split(' ')).astype(int)

n, S = D.size, 1000
mu, sigma = np.mean(D), np.std(D)

# generate posterior samples
np.random.seed(0)
rep = sigma * np.random.randn(S, n) + mu

plt.figure(figsize=(6,5))
plt.hist(D, 20)
plt.title('Newcombs Speed of Light Data', fontsize=14 )
pml.savefig('newcomb-truth2.pdf', dpi=300)
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(6, 6))
fig.suptitle('Posterior Samples', y=1.05, fontsize=14)

for i in range(9):
  plt.subplot(3, 3, i+1)
  plt.hist(rep[i,:])
  plt.tight_layout()

pml.savefig('newcomb-synth2.pdf')
plt.show()

test_val = np.array([np.min(rep[s,:]) for s in range(S)])
test_val_true = np.min(D)
plot_posterior(test_val, test_val_true, f'Posterior of min(x), true min={test_val_true}', 'newcomb-test-stat2')

test_val = np.array([np.var(rep[s,:]) for s in range(S)])
test_val_true = np.var(D)
plot_posterior(test_val, test_val_true, f'Posterior of var(x), true var={np.round(test_val_true, 2)}', 'newcomb-histo-var')