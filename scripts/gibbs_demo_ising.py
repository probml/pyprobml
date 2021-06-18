# -*- coding: utf-8 -*-
"""
Author : Ming Liang Ang
Based on : https://github.com/probml/pmtk3/blob/master/demos/gibbsDemoIsing.m
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pyprobml_utils as pml

pixelX = 100
pixelY = 100

def sigmoid(x):
  return 1/(1+np.exp(-x))

def energy(ix, iy, X, J):
  wi = 0
  if iy > 0:
    wi += X[iy-1, ix]
  if iy < pixelY-1:
    wi += X[iy+1, ix]
  if ix > 0:
    wi += X[iy, ix-1]
  if ix<pixelX-1:
    wi += X[iy, ix+1]
  return 2*J*wi

def gibbs(rng, pixelX, pixelY, J, niter=50000):
  X = ( 2 * ( rng.random( (pixelX, pixelY) )> 0.5 ) - 1 )
  for iter in tqdm(range(niter)):

    ix = np.ceil((pixelX-1) * rng.random(1)).astype(int)  
    iy = np.ceil((pixelY-1) * rng.random(1)).astype(int)
    
    e = energy(ix, iy, X, J)
  
    if rng.random(1) < sigmoid(e):
      X[ iy, ix] = 1
    else:
      X[ iy, ix] = -1
  return X

temps = [5, 2.5, 0.1]
seed = 12

fig, axs = plt.subplots(1, len(temps), figsize=(8, 8))
rng = np.random.default_rng(seed)
for t, T in enumerate(temps):
  J = 1/T
  axs[t].imshow(gibbs(rng, pixelX, pixelY, J), cmap="Greys" )
  axs[t].set_title(f"Trial {t+1} temp {T}")
pml.savefig('gibbsDemoIsing.pdf')
plt.show()
