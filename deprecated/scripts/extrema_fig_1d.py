# Plot local minimum and maximum in 1d

#https://nbviewer.jupyter.org/github/entiretydotai/Meetup-Content/blob/master/Neural_Network/7_Optimizers.ipynb

import superimport

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import os
from matplotlib import colors as mcolors
import pyprobml_utils as pml


def f(x):
    return x * np.sin(-np.pi*x)

a = np.arange(-1,3,0.01)
plt.annotate('local minimum', xy=(0.7, -0.55), xytext=(0.1, -2.0),
            arrowprops=dict(facecolor='black'))

plt.annotate('Global minimum', xy=(2.5, -2.5), xytext=(0.1, -2.5),
            arrowprops=dict(facecolor='black'))

plt.plot(a,f(a));
pml.save_fig("extrema_fig_1d.pdf")
plt.show()
