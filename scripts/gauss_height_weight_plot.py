
# Gaussian in 2d fit to height/weight data
# Author: Duane Rich
# Based on matlab code by Kevin Murphy
# https://github.com/probml/pmtk3/blob/master/demos/gaussHeightWeight.m


import numpy as np
import matplotlib.pyplot as plt
import os
#figdir = os.path.join(os.environ["PYPROBML"], "figures")
#datadir = os.path.join(os.environ["PYPROBML"], "data", "heightWeight")
figdir = '../figures'
datadir = '../data/heightWeight'
def save_fig(fname): plt.savefig(os.path.join(figdir, fname))

from matplotlib.patches import Ellipse

import scipy.io
dataAll = scipy.io.loadmat(os.path.join(datadir, "heightWeight.mat"))
data = dataAll['heightWeightData']

sex = data[:, 0]
x = data[:, 1]
y = data[:, 2]
male_arg = (sex == 1)
female_arg = (sex == 2)
x_male = x[male_arg]
y_male = y[male_arg]
x_female = x[female_arg]
y_female = y[female_arg]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_male, y_male, 'bx')
ax.plot(x_female, y_female, 'ro')
save_fig('heightWeightScatter.pdf')
plt.show()

def draw_ell(cov, xy, color):
    u, v = np.linalg.eigh(cov)
    angle = np.arctan2(v[0][1], v[0][0])
    angle = (180 * angle / np.pi)
    # here we time u2 with 5, assume 95% are in this ellipse~
    u2 = 5 * (u ** 0.5)
    e = Ellipse(xy, u2[0], u2[1], angle)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_facecolor('none')
    e.set_edgecolor(color)

cov_matrix1 = np.cov(np.vstack([x_female.ravel(), y_female.ravel()]))
xy1 = (np.mean(x_female), np.mean(y_female))
cov_matrix2 = np.cov(np.vstack([x_male.ravel(), y_male.ravel()]))
xy2 = (np.mean(x_male), np.mean(y_male))
draw_ell(cov_matrix1, xy1, 'r')
draw_ell(cov_matrix2, xy2, 'b')
save_fig('heightWeightScatterCov.pdf')
plt.show()
