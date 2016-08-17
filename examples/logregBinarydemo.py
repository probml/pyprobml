#!/usr/bin/env python

# Fit the following models to SAT scores:
#   - Logistic Regression
#   - Quadratic Logistic Regression
#   - RBF Logistic Regression
#   - KNN with 10 nearest neighbors

import matplotlib.pyplot as pl
import numpy as np
import utils.util as util
from scipy.special import logit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.neighbors import KNeighborsClassifier

def genMultinomialData(num_instances, num_classes, num_vars):
  num_example_points = 3

  np.random.seed(234)
  example_points = np.random.randn(num_classes * num_example_points, num_vars)

  np.random.seed(234)
  X = 2*np.random.rand(num_instances, num_vars)-1

  y = np.zeros((num_instances, 1))

  # Now 'classify' each instance by its nearest neighbor.
  for i in range(1, num_instances):
    # Take the i'th example and find the closest sample.
    dist = np.linalg.norm((
             np.tile(X[i,:], (num_classes * num_example_points, 1)) -
             example_points), axis=1)
    min_index = np.argmin(dist)
    y[i, 0] = (min_index % num_classes) + 1

  return X,y

def plotScatter(X0, X1, y):
  for x0, x1, cls in zip(X0, X1, y):
    color = 'blue' if cls == 1 else 'red'
    marker = 'x' if cls == 1 else 'o'
    pl.scatter(x0, x1, marker=marker, color=color)

X,y = genMultinomialData(100, 2, 2)

models = [LogisticRegressionCV(),
          LogisticRegressionCV(),
          LogisticRegressionCV(),
          KNeighborsClassifier(n_neighbors=10)]
kernels = [lambda X0, X1: X0, # No Kernel
           lambda X0, X1: polynomial_kernel(X0, X1, degree=2),
           lambda X0, X1: rbf_kernel(X0, X1, gamma=50), # sigma = .1
           lambda X0, X1: X0]
names = ['Linear Logistic Regression', 
         'Quadratic Logistic Regression', 
         'RBF Logistic Regression',
         'KNN with K=10']
file_names = ['Linear', 'Quad', 'Rbf', 'KNN10']

for i in range(len(models)):
  transX = kernels[i](X, X)
  model = models[i].fit(transX, y)
  
  xx, yy = np.meshgrid(np.linspace(-1, 1, 250), np.linspace(-1, 1, 250))
  Z = model.predict(kernels[i](np.c_[xx.ravel(), yy.ravel()], X)).reshape(xx.shape)
  pl.pcolormesh(xx, yy, Z, cmap=pl.cm.coolwarm)
  plotScatter(X[:, 0], X[:, 1], y)
  pl.title(names[i])
  pl.savefig('logregBinary%sBoundary' % file_names[i])
  pl.show()
  
  Z = model.predict_proba(kernels[i](np.c_[xx.ravel(), yy.ravel()], X))[:,2].reshape(xx.shape)
  pl.pcolormesh(xx, yy, Z, cmap=pl.cm.coolwarm)
  pl.colorbar()
  pl.title('Prob Class 1')
  pl.savefig('logregBinary%sProbClass1' % file_names[i])
  pl.show()
