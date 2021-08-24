
# Gaussian in 2d fit to height/weight data
# Author: Duane Rich
# Based on matlab code by Kevin Murphy
# https://github.com/probml/pmtk3/blob/master/demos/gaussHeightWeight.m


import superimport

import numpy as np
import matplotlib.pyplot as plt
import pyprobml_utils as pml
from matplotlib.patches import Ellipse

from scipy.io import loadmat
import requests
from io import BytesIO

#dataAll = scipy.io.loadmat(os.path.join(datadir, "heightWeight.mat"))
#data = dataAll['heightWeightData']

url = 'https://raw.githubusercontent.com/probml/probml-data/main/data/heightWeight/heightWeight.mat'
response = requests.get(url)
rawdata = BytesIO(response.content)
dataAll = loadmat(rawdata)
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

fig, ax  = plt.subplots()
ax.plot(x_male, y_male, 'bx')
ax.plot(x_female, y_female, 'ro')
pml.savefig('heightWeightScatter.pdf')
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

fig, ax  = plt.subplots()
ax.plot(x_male, y_male, 'bx')
ax.plot(x_female, y_female, 'ro')
draw_ell(cov_matrix1, xy1, 'r')
draw_ell(cov_matrix2, xy2, 'b')
pml.savefig('heightWeightScatterCov.pdf')
plt.show()
