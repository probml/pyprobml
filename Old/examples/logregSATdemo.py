#!/usr/bin/env python

# Fit logistic model to SAT scores.

import matplotlib.pyplot as plt
import numpy as np
from utils import util
from scipy.special import logit
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

data = util.load_mat('sat/sat.mat')
sat = data['sat']

# 3rd column contains SAT scores
X, y = sat[:,3], sat[:,0]
X = X.reshape((len(X), 1))
#y = y.reshape((len(X), 1))

#logistic = LogisticRegressionCV() # by default, cv=None, C=10
logistic = LogisticRegression(C=1e9) # turn off regularization
model = logistic.fit(X, y)

xtest = np.linspace(400, 700, 100)
xtest = xtest.reshape((len(xtest), 1))
# Solve for the decision boundary
a = model.coef_; b = model.intercept_;
threshold = (logit(0.5) - b)/a;

fig, ax = plt.subplots()
#plt.axis([450, 655, -.05, 1.05])
plt.axis([400, 700, -.05, 1.05])
plt.plot(X, y, 'ko')
#plt.plot(X, model.predict_proba(X)[:,1], 'ro')
plt.plot(xtest, model.predict_proba(xtest)[:,1], 'r-')
plt.plot(525, 0, 'bx', linewidth=2, markersize=14);
plt.plot(525, 1, 'bx', linewidth=2, markersize=14);
l = plt.axvline(threshold, linewidth=3, color='k')


plt.xlabel('SAT score')
plt.ylabel('Prob. pass class')
plt.title('Logistic regression on SAT data, threshold = %2.1f' % threshold)
plt.savefig('figures/logregSATdemoPython.pdf')
plt.draw()

# Fit linear regression model
model = LinearRegression()
model = model.fit(X, y)

fig, ax = plt.subplots()
plt.axis([400, 700, -3, 3])
#plt.axis([450, 655, -.05, 1.05])
plt.plot(X, y, 'ko')
#plt.plot(xtest, model.predict(xtest), 'ko')
yhat = model.predict(xtest)
plt.plot(xtest, yhat, 'b-')

plt.xlabel('SAT score')
plt.ylabel('Predicted output')
plt.title('Linear regression on SAT data')
plt.savefig('figures/linregSATdemoPython.pdf')
plt.draw()

plt.show()
