#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from utils import util
from scipy.special import logit
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import PolynomialFeatures

def create_xor_data(N):
    #np.random.seed(234)
    np.random.RandomState(0)
    C = 0.01*np.eye(2)
    Gs = [mvn(mean=[0.5,0.5], cov=C),
          mvn(mean=[-0.5,-0.5], cov=C),
          mvn(mean=[0.5,-0.5], cov=C),
          mvn(mean=[-0.5,0.5], cov=C)]
    X = np.concatenate([G.rvs(size=N) for G in Gs])
    y = np.concatenate((np.zeros(2*N), np.ones(2*N)))
    return X,y


def plotScatter(X0, X1, y):
  for x0, x1, cls in zip(X0, X1, y):
    #color = 'blue' if cls == 1 else 'red'
    color = 'red' if cls == 1 else 'blue'
    marker = 'x' if cls == 1 else 'o'
    plt.scatter(x0, x1, marker=marker, color=color)

X,y = create_xor_data(10)


transformers = [PolynomialFeatures(1), # no-op
               PolynomialFeatures(2),
               PolynomialFeatures(1),
                PolynomialFeatures(1),
               PolynomialFeatures(1)]

models = [LogisticRegression(C=1.0),
          LogisticRegression(C=1.0),
          LogisticRegression(C=1.0),
          LogisticRegression(C=1.0),
          LogisticRegression(C=1.0),]
  
kernels = [lambda X0, X1: X0, # No Kernel
           lambda X0, X1: X0, # No Kernel
           lambda X0, X1: linear_kernel(X0, X1),
           lambda X0, X1: polynomial_kernel(X0, X1, degree=2),
           lambda X0, X1: rbf_kernel(X0, X1, gamma=15)]

names = ['Linear Logistic Regression', 
         'Quadratic Logistic Regression',
         'Linear kernel',
         'Quadratic kernel',
         'RBF Kernel']

file_names = ['Linear', 'Quad', 'LinearKernel', 'QuadKernel', 'Rbf']
# pdf image files are very big (1MB), png is ~24kb
#file_type = '.pdf'
file_type = '.png'

for i in range(len(models)):
  transformer = transformers[i]
  XX = transformer.fit_transform(X)[:,1:] # skip the first column of 1s
  transX = kernels[i](XX, XX)
  model = models[i].fit(transX, y)
  print('experiment %d' % (i))
  #print(model.Cs_)
  #print(model.C_)
  #print(model.scores_)
  
  xx, yy = np.meshgrid(np.linspace(-1, 1, 250), np.linspace(-1, 1, 250))
  grid = np.c_[xx.ravel(), yy.ravel()]
  grid2 = transformer.transform(grid)[:,1:]
  Z = model.predict(kernels[i](grid2, XX)).reshape(xx.shape)
  fig, ax = plt.subplots()
  plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm)
  plotScatter(X[:, 0], X[:, 1], y)
  plt.title(names[i])
  fname = 'figures/logregXor%sBoundary%s' % (file_names[i], file_type)
  print(fname)
  plt.savefig(fname, dpi=600)
  
  
  plt.draw()

  if True:
      Z = model.predict_proba(kernels[i](grid2, XX))[:,1].reshape(xx.shape)
      fig, ax = plt.subplots()
      plt.pcolormesh(xx, yy, Z, cmap=plt.cm.coolwarm)
      plt.colorbar()
      plt.title('Prob Class 1')
      plt.savefig('figures/logregXor%sProbClass1%s' % (file_names[i], file_type))
      plt.draw()

plt.show()
