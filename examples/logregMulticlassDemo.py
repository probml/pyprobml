#!/usr/bin/env python

# Fit logistic regression models to 3 classs 2d data.

import matplotlib.pyplot as plt
import numpy as np
from utils import util
from scipy.special import logit
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import multivariate_normal as mvn

def genMultinomialData(num_instances, num_classes, num_vars):
  num_example_points = 3

  np.random.seed(234)
  example_points = np.random.randn(num_classes * num_example_points, num_vars)

  np.random.seed(234)
  X = 2*np.random.rand(num_instances, num_vars)-1

  #y = np.zeros((num_instances, 1))
  y = np.zeros(num_instances)

  # Now 'classify' each instance by its nearest neighbor.
  for i in range(1, num_instances):
    # Take the i'th example and find the closest sample.
    dist = np.linalg.norm((
             np.tile(X[i,:], (num_classes * num_example_points, 1)) -
             example_points), axis=1)
    min_index = np.argmin(dist)
    #y[i, 0] = (min_index % num_classes) + 1
    y[i] = (min_index % num_classes) + 1

  return X,y

def create_data(N):
    np.random.seed(234)
    #np.random.RandomState(0)
    C = 0.01*np.eye(2)
    Gs = [mvn(mean=[0.5,0.5], cov=C),
          mvn(mean=[-0.5,-0.5], cov=C),
          mvn(mean=[0.5,-0.5], cov=C),
          mvn(mean=[-0.5,0.5], cov=C),
          mvn(mean=[0,0], cov=C)]
    X = np.concatenate([G.rvs(size=N) for G in Gs])
    y = np.concatenate((1*np.ones(N), 1*np.ones(N),
                        2*np.ones(N), 2*np.ones(N),
                        3*np.ones(N)))
    return X,y


def plotScatter(X0, X1, y):
  for x0, x1, cls in zip(X0, X1, y):
    colors = ['blue', 'black', 'red']
    markers = ['x', 'o', '*']
    color = colors[int(cls)-1]
    marker = markers[int(cls)-1]
    plt.scatter(x0, x1, marker=marker, color=color)

if False:    
    X,y = genMultinomialData(100, 3, 2)
    X = X[1:,:] # skip first entry
    y = y[1:]
else:
    X, y = create_data(100)

print X
print X.shape
print y
print y.shape
#exit()

models = [LogisticRegression(C=1.0),
            LogisticRegression(C=1.0)]

transformers = [PolynomialFeatures(1), # no-op
               PolynomialFeatures(2)]

names = ['Linear Logistic Regression', 
         'Quadratic Logistic Regression']
file_names = ['Linear', 'Quad']

# pdf image files are very big (1MB), png is ~24kb
#file_type = '.pdf'
file_type = '.png'

for i in range(len(models)):
  transformer = transformers[i]
  XX = transformer.fit_transform(X)[:,1:] # skip the first column of 1s
  model = models[i].fit(XX, y)
  print('experiment %d' % (i))

  xx, yy = np.meshgrid(np.linspace(-1, 1, 250), np.linspace(-1, 1, 250))
  grid = np.c_[xx.ravel(), yy.ravel()]
  grid2 = transformer.transform(grid)[:,1:]
  Z = model.predict(grid2).reshape(xx.shape)
  fig, ax = plt.subplots()
  plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm)
  plotScatter(X[:, 0], X[:, 1], y)
  #plt.scatter(X[:,0], X[:,1], y)
  plt.title(names[i])

  plt.savefig('figures/logregMulti%sBoundary%s' % (file_names[i], file_type))
  plt.draw()

plt.show()
