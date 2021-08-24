# -*- coding: utf-8 -*-

# V. Kuleshov and P. S. Liang, “Calibrated Structured Prediction,” in NIPS, 2015, pp. 3474–3482 [Online]. 
#Available: http://papers.nips.cc/paper/5658-calibrated-structured-prediction.pdf

import superimport

import numpy as np
import matplotlib.pyplot as plt
import os

eps = 0.01 # we make this non-zero for plotting purposes
ptrue = [eps, 1, 0.5] # expect[y|x]
Fcal = [0.5, 0.5, 0.5]
Funcal = [0.2, 0.8, 0.4]
Fbal = [eps, 0.75, 0.75]

#https://matplotlib.org/examples/api/barchart_demo.html

width = 0.2

fig, ax = plt.subplots()
X = np.arange(3) 
bar_true = ax.bar(X, ptrue, width, color='r')
bar_cal = ax.bar(X+width,  Fcal, width, color='g')
bar_uncal = ax.bar(X+2*width, Funcal, width, color='b')
bar_bal = ax.bar(X+3*width, Fbal, width, color='k')

ax.set_ylabel('probability')
ax.set_xticks(X+width)
ax.set_xticklabels(X)

ax.legend((bar_true[0], bar_cal[0], bar_uncal[0], bar_bal[0]),
    ('true', 'cal', 'uncal', 'bal'))

plt.show()
plt.savefig(os.path.join('figures', 'calibration'))

### Plot error

eps = 0 
ptrue = np.array([eps, 1, 0.5])
Fcal = np.array([0.5, 0.5, 0.5])
Funcal = np.array([0.2, 0.8, 0.4])
Fbal = np.array([eps, 0.75, 0.75])

err_cal = (ptrue - Fcal)
err_uncal = (ptrue - Funcal)
err_bal = (ptrue - Fbal)
fig, ax = plt.subplots()

X = np.arange(3) 
bar_cal = ax.bar(X+width,  err_cal, width, color='g')
bar_uncal = ax.bar(X+2*width, err_uncal, width, color='b')
bar_bal = ax.bar(X+3*width, err_bal, width, color='k')

ax.set_ylabel('error')
ax.set_xticks(X+width)
ax.set_xticklabels(X)

ax.legend((bar_cal[0], bar_uncal[0], bar_bal[0]),
    ('cal', 'uncal', 'bal'))

plt.show()
plt.savefig(os.path.join('figures', 'calibration_err'))
