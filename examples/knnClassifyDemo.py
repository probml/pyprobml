#!/usr/bin/env python

import os
import matplotlib.pyplot as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cross_validation import cross_val_score
from utils.util import DATA_DIR


def load_data():
  """Since the knnClassify3c.mat is the matlab v7.3 or later file
  we have to load data from txt"""
  train_file = os.path.join(DATA_DIR, 'knnClassify3cTrain.txt')
  test_file = os.path.join(DATA_DIR, 'knnClassify3cTest.txt')
  train = np.loadtxt(train_file,
                     dtype=[('x_train', ('f8', 2)),
                            ('y_train', ('f8', 1))])
  test = np.loadtxt(test_file,
                    dtype=[('x_test', ('f8', 2)),
                           ('y_test', ('f8', 1))])
  return train['x_train'], train['y_train'], test['x_test'], test['y_test']

x_train, y_train, x_test, y_test = load_data()

#plot train fig
pl.figure()
y_unique = np.unique(y_train)
markers = '*x+'
colors = 'bgr'
for i in range(len(y_unique)):
  pl.scatter(x_train[y_train == y_unique[i], 0],
             x_train[y_train == y_unique[i], 1],
             marker=markers[i],
             c=colors[i])
pl.savefig('knnClassifyDemo_1.png')

#plot test fig
pl.figure()
for i in range(len(y_unique)):
  pl.scatter(x_test[y_test == y_unique[i], 0],
             x_test[y_test == y_unique[i], 1],
             marker=markers[i],
             c=colors[i])
pl.savefig('knnClassifyDemo_2.png')

x = np.linspace(np.min(x_test[:, 0]), np.max(x_test[:, 0]), 200)
y = np.linspace(np.min(x_test[:, 1]), np.max(x_test[:, 1]), 200)
xx, yy = np.meshgrid(x, y)
xy = np.c_[xx.ravel(), yy.ravel()]

# use the knn model to predict
for k in [1, 5, 10]:
  knn = KNN(n_neighbors=k)
  knn.fit(x_train, y_train)
  pl.figure()
  y_predicted = knn.predict(xy)
  pl.pcolormesh(y_predicted.reshape(200, 200))
  pl.title('k=%s' % (k))
  pl.savefig('knnClassifyDemo_k%s.png' % (k))

#plot train err and test err with different k
ks = [1, 5, 10, 20, 50, 100, 120]
train_errs = []
test_errs = []
for k in ks:
  knn = KNN(n_neighbors=k)
  knn.fit(x_train, y_train)
  train_errs.append(1 - knn.score(x_train, y_train))
  test_errs.append(1 - knn.score(x_test, y_test))
pl.figure()
pl.plot(ks, train_errs, 'bs:', label='train')
pl.plot(ks, test_errs, 'rx-', label='test')
pl.legend()
pl.xlabel('k')
pl.ylabel('misclassification rate')
pl.savefig('knnClassifyDemo_4.png')

#cross_validate
scores = []
for k in ks:
    knn = KNN(n_neighbors=k)
    score = cross_val_score(knn, x_train, y_train, cv=5)
    scores.append(1 - score.mean())
pl.figure()
pl.plot(ks, scores, 'ko-')
min_k = ks[np.argmin(scores)]
pl.plot([min_k, min_k], [0, 1.0], 'b-')
pl.xlabel('k')
pl.ylabel('misclassification rate')
pl.title('5-fold cross validation, n-train = 200')

#draw hot-map to show the probability of different class
knn = KNN(n_neighbors=10)
knn.fit(x_train, y_train)
xy_predic = knn.predict_proba(xy)
levels = np.arange(0, 1.01, 0.1)
for i in range(3):
    pl.figure()
    pl.contourf(xy_predic[:, i].ravel().reshape(200, 200), levels)
    pl.colorbar()
    pl.title('p(y=%s | data, k=10)' % (i))
    pl.savefig('knnClassifyDemo_hotmap_%s.png' % (i))
pl.show()
