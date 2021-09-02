# Cosine annealing learning rate schedule
# https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/

import superimport

#from matplotlib import pyplot
from math import pi
from math import cos
from math import floor
 
import numpy as np
import matplotlib.pyplot as plt
import os


figdir = "../figures"
def save_fig(fname):
    if figdir: plt.savefig(os.path.join(figdir, fname))
    
# cosine annealing learning rate schedule
def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
	epochs_per_cycle = floor(n_epochs/n_cycles)
	cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
	return lrate_max/2 * (cos(cos_inner) + 1)
 
# create learning rate series
n_epochs = 100
n_cycles = 5
lrate_max = 0.01
series = [cosine_annealing(i, n_epochs, n_cycles, lrate_max) for i in range(n_epochs)]
# plot series
plt.figure()
plt.plot(series)
fname = 'lrschedule_cosine_annealing.pdf'
save_fig(fname)
plt.show()

