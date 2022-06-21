# -*- coding: utf-8 -*-

import superimport

from math import floor
from numpy import arange
from scipy.stats import gamma
import matplotlib.pyplot as plt

alpha = 5.7
lam = 2

k = floor(alpha)
M = gamma.pdf(alpha-k, alpha, scale = 1/lam) / gamma.pdf(alpha-k, k, scale = 1/(lam-1))
xs = arange(0,10,0.01)

fig, ax = plt.subplots()
ax.set_xlim(0,10)
ax.set_ylim(0,1.4)
ax.plot(xs, gamma.pdf(xs,alpha,scale = 1/lam), 'b-',linewidth=3, label='target p(x)')
ax.plot(xs, M*gamma.pdf(xs,k,scale = 1/(lam-1)), 'r:',linewidth = 3, label='comparison function Mq(x)')
ax.legend(loc='upper right')
plt.show()
