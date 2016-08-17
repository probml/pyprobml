#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
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


def L2loss(yhat, ytest):
    ntest = ytest.size
    sqerr = np.power(yhat - ytest, 2)
    mse = np.mean(sqerr)
    stderr = np.std(sqerr) / np.sqrt(ntest)
    return (mse, stderr, np.sqrt(sqerr))
    

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
ntest = ytest.size


### Process data

methods=[LinearRegression(),  RidgeCV(cv=3), LassoCV()]
method_names = ["LS", "Ridge", "Lasso"]

# Hash table to store parameters and performance, indexed by method name
coefHt = {}
mseHt = {}
stderrHt = {}
errorsHt = {}

for i,method in enumerate(methods):
  name = method_names[i]
  clf = method
  model = clf.fit(Xtrain, ytrain.ravel())
  coef = np.append(model.intercept_, model.coef_)
  coefHt[name] = coef
  yhat = model.predict(Xtest)
  #mse = mean_squared_error(yhat, ytest)
  (mseHt[name], stderrHt[name], errorsHt[name]) = L2loss(yhat, ytest) 


method_names.append("Subset")
name = "Subset"
clf = LinearRegression()
subset = _best_subset_cv(clf, Xtrain, ytrain, cv=3)
model = clf.fit(Xtrain[:, subset], ytrain)

ndims = Xtrain.shape[1]
coef = np.zeros(ndims)
subset_ndx = np.asarray(subset) # convert from tuple to array
coef[subset_ndx] = model.coef_
coef = np.append(model.intercept_, coef)
coefHt[name] = coef

yhat = model.predict(Xtest[:, subset])
#mse = mean_squared_error(yhat, ytest)
(mseHt[name], stderrHt[name], errorsHt[name]) = L2loss(yhat, ytest)

print method_names
print mseHt
print coefHt

### Pretty print the results in latex format
coef_names = ["intercept", "lcalvol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
method_names = ['LS', 'Subset', 'Ridge', 'Lasso'] # Choose desired ordering of methods

print " Term &" , " & ".join(method_names)
for coef_ndx, coef_name in enumerate(coef_names):
    str_list = []
    for method_name in method_names:
        method_coefs = coefHt[method_name]
        coef = method_coefs[coef_ndx]
        str_list.append(_format(coef))
    str_list.insert(0, coef_name)
    print " & ".join([str(s) for s in str_list]), "\\\\"

str_list = []
for method_name in method_names:
    str_list.append(_format(mseHt[method_name]))
str_list.insert(0, "Test error")
print " & ".join([str(s) for s in str_list]), "\\\\"

str_list = []
for method_name in method_names:
    str_list.append(_format(stderrHt[method_name]))
str_list.insert(0, "Std error")
print " & ".join([str(s) for s in str_list]), "\\\\"

# Boxplot of errors for each method
nmethods = len(method_names)
ntest  = np.shape(ytest)[0]
errorsMatrix = np.zeros((ntest, nmethods))
for i in range(0, nmethods):
    method_name = method_names[i]
    errorsMatrix[:,i] = errorsHt[method_name]
plt.boxplot(errorsMatrix)
ax = plt.gca()
ax.set_xticklabels(method_names)
#fname = '/Users/kpmurphy/GDrive/Backup/MLbook/book2.0/Figures/pdfFigures/prostateBoxplot.pdf'
#plt.savefig(fname, bbox_inches='tight')
