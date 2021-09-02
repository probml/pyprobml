# Plot saddle point in 2d
#https://nbviewer.jupyter.org/github/entiretydotai/Meetup-Content/blob/master/Neural_Network/7_Optimizers.ipynb

import superimport

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import os
from matplotlib import colors as mcolors
import pyprobml_utils as pml

fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = X**2 - Y**2

ax.text2D(0,0,'*',fontdict={'ha': 'center', 'va': 'center', 'family': 'sans-serif','fontweight': 'bold',"fontsize":20})
ax.text2D(0,0.01,'Saddle Point',fontdict={'family': 'sans-serif','fontweight': 'bold'})
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel(r'$Z$')
ax.plot_surface(X, Y, Z) #color="red");
pml.savefig("saddle.pdf")
plt.show()
