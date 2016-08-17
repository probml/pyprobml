#!/usr/bin/env python3

import csv
import numpy as np
import pandas as pd
import sklearn
from itertools import chain, combinations
from scipy.stats import linregress
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale

pd.set_option('display.max_columns', 160)
pd.set_option('display.width', 1000)

### Helper functions

def _format(s):
  return '{0:.3f}'.format(s)

# Normalize each column to have a mean of 0, std dev of 1. Use ddof=1
# to get consistent results with R.
def _scale(X):
  for column in X.T:
    mu = np.mean(column)
    sigma = np.std(column, ddof=1)
    column -= mu
    column /= sigma
  return X

# Returns the subset of features that give the smallest Least Squares error.
def _best_subset_cv(model, X, y, cv=3):
  n_features = X.shape[1]
  subsets = chain.from_iterable(combinations(range(n_features), k+1) for k in range(n_features + 1))
  best_score = -np.inf
  best_subset = None
  for subset in subsets:
      score = cross_val_score(model, X[:, subset], y, cv=cv).mean()
      if score > best_score:
          best_score, best_subset = score, subset

  return best_subset


#### Get data

url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
#filename = '/Users/kpmurphy/github/pmtk3/prostate.csv'
df = pd.read_csv(url, sep='\t', header=0)
data = df.values[:,1:] # skip the column of indices
(nr, nc) = data.shape
istrain_str = data[:, 9]
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)
y = np.double(data[:,8])
X = np.double(data[:,0:8]) # note 0:8 actually means columns 0 to 7 inclusive!

#Xscaled = _scale(X)
Xscaled = sklearn.preprocessing.scale(X)

Xtrain = Xscaled[istrain, :]
Xtest = Xscaled[istest, :]
ytrain = y[istrain]
ytest = y[istest]

#X = util.load_mat('prostate')
# Hack to use the correct dataset.
#X['Xtest'][8][1] = 3.804438
# Rescale all data at once.
#Xscaled = _scale(np.append(X['Xtrain'], X['Xtest'], axis=0))
#Xtrain = Xscaled[0:67,:]
#Xtest = Xscaled[67:,:]
#ytrain = X['ytrain']
#ytest = X['ytest']

### Process data

methods=[LinearRegression(),  RidgeCV(cv=3), LassoCV()]
method_names = ["LS", "Ridge", "Lasso"]
intercepts=["Intercept"]
coefficients=[["lcalvol"], ["lweight"], ["age"], ["lbph"], ["svi"], ["lcp"], ["gleason"], ["pgg45"]]
MSEs=["Test Error"]
#SEs=["Standard Error"]

coefHt = {}
mseHt = {}

for i,method in enumerate(methods):
  name = method_names[i]
  clf = method
  model = clf.fit(Xtrain, ytrain.ravel())
  intercepts.append(_format(model.intercept_))

  for i,coef in enumerate(model.coef_):
    coefficients[i].append(_format(coef))

  coefHt[name] = model.coef_
  yhat = model.predict(Xtest)
  mse = mean_squared_error(yhat, ytest)
  MSEs.append(_format(mse))
  mseHt[name] = mse


method_names.append("Subset")
clf = LinearRegression()
subset = _best_subset_cv(clf, Xtrain, ytrain, cv=3)
model = clf.fit(Xtrain[:, subset], ytrain)

for i in range(Xtrain.shape[1]):
  coefficients[i].append(0.00)
for i,coef in enumerate(model.coef_.ravel()):
  coefficients[i][-1] = _format(coef)

#intercepts.append(_format(model.intercept_[0]))
intercepts.append(_format(model.intercept_))
MSEs.append(_format(mean_squared_error(model.predict(Xtest[:, subset]), ytest)))

# Write CSV
CSV=[method_names, intercepts]
CSV+=coefficients
CSV.append(MSEs)

with open("prostateComparison.txt", "wb") as f:
  writer = csv.writer(f, delimiter='&')
  writer.writerows(CSV)

print method_names, MSEs

print mseHt
print coefHt


