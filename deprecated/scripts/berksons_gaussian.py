# Illustrate Explaining Away  (Berskson's Paraodx) for a gaussian DAG
# Based on https://ff13.fastforwardlabs.com/

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import pyprobml_utils as pml
import seaborn as sns

np.random.seed(0)

def x():
  return -5 + randn()

def y():
  return 5 + randn()

def z(x, y):
  return x + y + randn()

def sample():
  x_ = x()
  y_ = y()
  z_ = z(x_, y_)
  return x_, y_, z_

np.random.seed(0)

nsamples = 8000
xs = -5 + randn(nsamples)
ys = 5 + randn(nsamples)
zs = xs + ys + randn(nsamples)


num_bins = 20
plt.figure(figsize=(6, 6))
_ = plt.hist(xs, num_bins, facecolor='#7FDAD9',  label='x')
_ = plt.hist(ys, num_bins, facecolor='#B5C5E2',label='y')
_ = plt.hist(zs, 2*num_bins, facecolor='#FAC9AC',alpha=0.8, label='z')
plt.ylabel('number of samples', fontsize=14)
plt.xlabel('value', fontsize=14)
plt.legend();
pml.savefig('berksons-hist.pdf', dpi=300)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(xs, ys, color='#00B6B5', alpha=0.1)
plt.ylabel('y', fontsize=14)
plt.xlabel('x', fontsize=14)
plt.xlim([-10, 0])
plt.ylim([0, 10])
pml.savefig('berksons-scatter.pdf', dpi=300)
plt.show()

indices = np.argwhere(zs > 2.5)
num_bins = 20
plt.figure(figsize=(6, 6))
_ = plt.hist(xs[indices], num_bins, facecolor='#7FDAD9',  label='x')
_ = plt.hist(ys[indices], num_bins, facecolor='#B5C5E2',label='y')
_ = plt.hist(zs[indices], 2*num_bins, facecolor='#FAC9AC',alpha=0.8, label='z')
plt.ylabel('number of samples', fontsize=14)
plt.xlabel('value', fontsize=14)
#plt.ylim([0, 300])
plt.legend();
pml.savefig('berksons-conditioned-hist.pdf', dpi=300)
plt.show()

plt.figure(figsize=(6, 6))
sns.regplot(x=xs[indices], y=ys[indices], ci=None, color='tab:orange', scatter_kws={'alpha':0.1, "color":'#00B6B5'})
plt.ylabel('y', fontsize=14)
plt.xlabel('x', fontsize=14)
plt.xlim([-10, 0])
plt.ylim([0, 10])
pml.savefig('berksons-conditioned-scatter.pdf', dpi=300)
plt.show()