# -*- coding: utf-8 -*-
# based on https://github.com/probml/pmtk3/blob/master/demos/arsEnvelope.m
# author : Ang Ming Liang

import superimport

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

xs = np.linspace(-1.5, 1.5, 50)
ps = norm.logpdf(xs)

dy_dx = lambda x : -x

fig, ax = plt.subplots()

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Fix the plot size
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-2.2,-0.8)

# Plotting the log-concave distribution
ax.plot(xs, ps, 'b-',linewidth=3)

# At x=-0.7
x1 = -0.7
ratio1 = dy_dx(x1)
ax.plot([x1 - 0.5, x1 + 0.5], [norm.logpdf(x1)-0.5*ratio1, norm.logpdf(x1)+0.5*ratio1], '-r', 'LineWidth', 3);
ax.plot([x1, x1], [norm.logpdf(x1),-2.2], '--r', 'LineWidth', 3);

# At x=0
x2 = 0
ratio2 = dy_dx(x2)
ax.plot([x2 - 0.5, x2 + 0.5], [norm.logpdf(x2)-0.5*ratio2, norm.logpdf(x2)+0.5*ratio2], '-r', 'LineWidth', 3);
ax.plot([x2, x2], [norm.logpdf(x2),-2.2], '--r', 'LineWidth', 3);

# At x=0.7
x3 = 0.7;
ratio3 = dy_dx(x3)
ax.plot([x3 - 0.5, x3 + 0.5], [norm.logpdf(x3)-0.5*ratio3, norm.logpdf(x3)+0.5*ratio3], '-r', 'LineWidth', 3);
ax.plot([x3, x3], [norm.logpdf(x3),-2.2], '--r', 'LineWidth', 3);

# Remove the x-axis and y-axis ticks
plt.xticks([])
plt.yticks([])

plt.show()

