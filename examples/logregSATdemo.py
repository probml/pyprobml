#!/usr/bin/env python

# Fit logistic model to SAT scores.

import matplotlib.pyplot as pl
import numpy as np
import utils.util as util
from scipy.special import logit
from sklearn.linear_model import LogisticRegressionCV

data = util.load_mat('sat/sat.mat')
sat = data['sat']

# 3rd column contains SAT scores
X, y = sat[:,3], sat[:,0]
X = X.reshape((len(X), 1))
y = y.reshape((len(X), 1))

logistic = LogisticRegressionCV()
print X
model = logistic.fit(X, y)

# Solve for the decision boundary
a = model.coef_; b = model.intercept_;
threshold = (logit(0.5) - b)/a;

pl.axis([450, 655, -.05, 1.05])
pl.plot(X, y, 'ko')
pl.plot(X, model.predict_proba(X)[:,1], 'ro')
pl.plot(525, 0, 'bx', linewidth=2, markersize=14);
pl.plot(525, 1, 'bx', linewidth=2, markersize=14);
l = pl.axvline(threshold, linewidth=3, color='k')


pl.xlabel('SAT score')
pl.ylabel('Prob. pass class')
pl.title('Logistic regression on SAT data, threshold = %2.1f' % threshold)
pl.savefig('logregSATdemo.png')
pl.show()
